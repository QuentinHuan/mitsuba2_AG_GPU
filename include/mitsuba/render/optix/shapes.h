#pragma once

#ifdef __CUDACC__
// List all shape's CUDA header files to be included in the PTX code generation.
// Those header files are located in '/mitsuba/src/shapes/optix/'
#include "cylinder.cuh"
#include "disk.cuh"
#include "mesh.cuh"
#include "rectangle.cuh"
#include "sphere.cuh"
#include "heightfield.cuh"
#else

#include <mitsuba/render/optix/common.h>
#include <mitsuba/render/optix_api.h>
#include <mitsuba/render/shape.h>

NAMESPACE_BEGIN(mitsuba)
/// List of the custom shapes supported by OptiX
static std::string custom_optix_shapes[] = {
    "Disk", "Rectangle", "Sphere", "Cylinder", "Heightfield",
};
static constexpr size_t custom_optix_shapes_count = std::size(custom_optix_shapes);

/// Retrieve index of custom shape descriptor in the list above for a given shape
template <typename Shape>
size_t get_shape_descr_idx(Shape *shape) {
    std::string name = shape->class_()->name();
    for (size_t i = 0; i < custom_optix_shapes_count; i++) {
        if (custom_optix_shapes[i] == name)
            return i;
    }
    Throw("Unexpected shape: %s. Couldn't be found in the "
          "'custom_optix_shapes' table.", name);
}

/// Stores two OptiXTraversables: one for the meshes and one for the custom shapes (e.g. sphere)
struct OptixAccelData {
    struct HandleData {
        OptixTraversableHandle handle = 0ull;
        void* buffer = nullptr;
        uint32_t count = 0u;
    };
    HandleData meshes;
    HandleData others;

    ~OptixAccelData() {
        if (meshes.buffer) cuda_free(meshes.buffer);
        if (others.buffer) cuda_free(others.buffer);
    }
};

/// Creates and appends the HitGroupSbtRecord for a given list of shapes
template <typename Shape>
void fill_hitgroup_records(std::vector<ref<Shape>> &shapes,
                           std::vector<HitGroupSbtRecord> &out_hitgroup_records,
                           const OptixProgramGroup *program_groups) {
    for (size_t i = 0; i < 2; i++) {
        for (Shape* shape: shapes) {
            // This trick allows meshes to be processed first
            if (i == !shape->is_mesh())
                shape->optix_fill_hitgroup_records(out_hitgroup_records, program_groups);
        }
    }
}

/**
 * \brief Build OptiX geometry acceleration structures (GAS) for a given list of shapes.
 *
 * Two different GAS will be created for the meshes and the custom shapes. Optix
 * handles to those GAS will be stored in an \ref OptixAccelData.
 */
template <typename Shape>
void build_gas(const OptixDeviceContext &context,
               const std::vector<ref<Shape>> &shapes,
               OptixAccelData& out_accel) {

    // Separate meshes and custom shapes
    std::vector<ref<Shape>> shape_meshes, shape_others;
    for (auto shape: shapes) {
        if (shape->is_mesh())           shape_meshes.push_back(shape);
        else if (!shape->is_instance()) shape_others.push_back(shape);
    }


    // Build a GAS given a subset of shape pointers
    auto build_single_gas = [&context](const std::vector<ref<Shape>> &shape_subset,
                                       OptixAccelData::HandleData &handle) {

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
        accel_options.motionOptions.numKeys = 0;
        if (handle.buffer) {
            cuda_free(handle.buffer);
            handle.handle = 0ull;
            handle.buffer = nullptr;
            handle.count = 0;
        }

        size_t shapes_count = shape_subset.size();

        if (shapes_count == 0)
            return;

        std::vector<OptixBuildInput> build_inputs(shapes_count);
        for (size_t i = 0; i < shapes_count; i++)
            shape_subset[i]->optix_build_input(build_inputs[i]);

        OptixAccelBufferSizes buffer_sizes;
        rt_check(optixAccelComputeMemoryUsage(
            context,
            &accel_options,
            build_inputs.data(),
            (unsigned int) shapes_count,
            &buffer_sizes
        ));

        void* d_temp_buffer = cuda_malloc(buffer_sizes.tempSizeInBytes);
        void* output_buffer = cuda_malloc(buffer_sizes.outputSizeInBytes + 8);

        OptixAccelEmitDesc emit_property = {};
        emit_property.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_property.result = (CUdeviceptr)((char*)output_buffer + buffer_sizes.outputSizeInBytes);

        OptixTraversableHandle accel;
        rt_check(optixAccelBuild(
            context,
            0,              // CUDA stream
            &accel_options,
            build_inputs.data(),
            (unsigned int) shapes_count, // num build inputs
            (CUdeviceptr)d_temp_buffer,
            buffer_sizes.tempSizeInBytes,
            (CUdeviceptr)output_buffer,
            buffer_sizes.outputSizeInBytes,
            &accel,
            &emit_property,  // emitted property list
            1                // num emitted properties
        ));

        cuda_free(d_temp_buffer);

        size_t compact_size;
        cuda_memcpy_from_device(&compact_size, (void*)emit_property.result, sizeof(size_t));
        if (compact_size < buffer_sizes.outputSizeInBytes) {
            void* compact_buffer = cuda_malloc(compact_size);
            // Use handle as input and output
            rt_check(optixAccelCompact(
                context,
                0, // CUDA stream
                accel,
                (CUdeviceptr)compact_buffer,
                compact_size,
                &accel
            ));
            cuda_free(output_buffer);
            output_buffer = compact_buffer;
        }

        handle.handle = accel;
        handle.buffer = output_buffer;
        handle.count = (uint32_t) shapes_count;
    };

    build_single_gas(shape_meshes, out_accel.meshes);
    build_single_gas(shape_others, out_accel.others);
}

/// Prepares and fills the \ref OptixInstance array associated with a given list of shapes.
template <typename Shape, typename Transform4f>
void prepare_ias(const OptixDeviceContext &context,
                       std::vector<ref<Shape>> &shapes,
                       uint32_t base_sbt_offset,
                       const OptixAccelData &accel,
                       uint32_t instance_id,
                       const Transform4f& transf,
                       std::vector<OptixInstance> &out_instances) {
    // Find all instances in the list of shapes
    std::vector<Shape*> instances;
    std::vector<uint32_t> instance_offsets;
    uint32_t offset = 0;
    for (Shape* shape: shapes) {
        if (shape->is_instance()) {
            instances.push_back(shape);
            instance_offsets.push_back(offset);
        }
        ++offset;
    }

    unsigned int sbt_offset = base_sbt_offset;

    float T[12] = { (float) transf.matrix(0, 0), (float) transf.matrix(0, 1),
                    (float) transf.matrix(0, 2), (float) transf.matrix(0, 3),
                    (float) transf.matrix(1, 0), (float) transf.matrix(1, 1),
                    (float) transf.matrix(1, 2), (float) transf.matrix(1, 3),
                    (float) transf.matrix(2, 0), (float) transf.matrix(2, 1),
                    (float) transf.matrix(2, 2), (float) transf.matrix(2, 3) };

    // Check whether transformation should be disabled on the IAS
    uint32_t flags = (transf == Transform4f())
                         ? OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM
                         : OPTIX_INSTANCE_FLAG_NONE;

    // Create an OptixInstance for the meshes if necessary
    if (accel.meshes.handle) {
        OptixInstance meshes_instance = {
            { T[0], T[1], T[2], T[3], T[4], T[5], T[6], T[7], T[8], T[9], T[10], T[11] },
            instance_id, sbt_offset, /* visibilityMask = */ 255,
            flags, accel.meshes.handle, /* pads = */ { 0, 0 }
        };
        out_instances.push_back(meshes_instance);
        sbt_offset += (unsigned int) accel.meshes.count;
    }

    // Create an OptixInstance for the custom shapes if necessary
    if (accel.others.handle) {
        OptixInstance others_instance = {
            { T[0], T[1], T[2], T[3], T[4], T[5], T[6], T[7], T[8], T[9], T[10], T[11] },
            instance_id, sbt_offset, /* visibilityMask = */ 255,
            flags, accel.others.handle, /* pads = */ { 0, 0 }
        };
        out_instances.push_back(others_instance);
    }

    // Apply the same process to every shape instances
    for (uint32_t i = 0; i < instances.size(); ++i)
        instances[i]->optix_prepare_ias(context, out_instances, instance_offsets[i], transf);
}

NAMESPACE_END(mitsuba)
#endif
