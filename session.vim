let SessionLoad = 1
if &cp | set nocp | endif
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/git/MITSUBA/mitsuba2_AG_GPU
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
let s:shortmess_save = &shortmess
if &shortmess =~ 'A'
  set shortmess=aoOA
else
  set shortmess=aoO
endif
badd +619 src/shapes/heightfield.cpp
badd +210 src/shapes/optix/heightfield.cuh
badd +39 src/librender/fermatNEE.cpp
badd +92 include/mitsuba/render/fermatNEE.h
badd +225 src/integrators/fermat_path.cpp
badd +337 include/mitsuba/render/shape.h
badd +2 include/mitsuba/core/object.h
badd +304 src/librender/scene.cpp
argglobal
%argdel
edit src/shapes/heightfield.cpp
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 135 + 135) / 270)
exe 'vert 2resize ' . ((&columns * 134 + 135) / 270)
argglobal
balt src/shapes/optix/heightfield.cuh
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=8
setlocal fdn=3
setlocal fen
silent! normal! zE
36,78fold
104,106fold
122,226fold
93,241fold
243,247fold
249,292fold
294,316fold
318,337fold
339,349fold
351,362fold
364,366fold
373,386fold
389,392fold
402,407fold
410,413fold
418,427fold
416,428fold
443,450fold
431,453fold
456,473fold
475,486fold
491,530fold
532,546fold
547,553fold
554,556fold
566,579fold
559,565fold
571,594fold
570,595fold
598,608fold
81,620fold
let &fdl = &fdl
81
normal! zo
532
normal! zo
547
normal! zo
554
normal! zo
570
normal! zo
571
normal! zo
598
normal! zo
let s:l = 619 - ((57 * winheight(0) + 31) / 63)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 619
normal! 05|
wincmd w
argglobal
if bufexists(fnamemodify("include/mitsuba/render/fermatNEE.h", ":p")) | buffer include/mitsuba/render/fermatNEE.h | else | edit include/mitsuba/render/fermatNEE.h | endif
balt src/librender/fermatNEE.cpp
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=8
setlocal fdn=3
setlocal fen
silent! normal! zE
31,33fold
35,38fold
40,42fold
44,47fold
49,59fold
15,60fold
73,79fold
80,86fold
63,166fold
let &fdl = &fdl
15
normal! zo
63
normal! zo
80
normal! zo
let s:l = 151 - ((45 * winheight(0) + 31) / 63)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 151
normal! 08|
wincmd w
exe 'vert 1resize ' . ((&columns * 135 + 135) / 270)
exe 'vert 2resize ' . ((&columns * 134 + 135) / 270)
tabnext 1
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20
let &shortmess = s:shortmess_save
let &winminheight = s:save_winminheight
let &winminwidth = s:save_winminwidth
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
