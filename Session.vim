let SessionLoad = 1
if &cp | set nocp | endif
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/Dropbox/Research/GPZIP
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
argglobal
%argdel
$argadd gpzip3.py
set stal=2
tabnew
tabnew
tabrewind
edit onoffgpf/OnOffSVGP.py
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 62 + 32) / 64)
exe 'vert 2resize ' . ((&columns * 1 + 32) / 64)
argglobal
let s:l = 27 - ((8 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
27
normal! 0
lcd ~/Dropbox/Research/GPZIP
wincmd w
argglobal
if bufexists("~/Dropbox/Research/GPZIP/zero-inflated-gp/onoffgpf/OnOffSVGP.py") | buffer ~/Dropbox/Research/GPZIP/zero-inflated-gp/onoffgpf/OnOffSVGP.py | else | edit ~/Dropbox/Research/GPZIP/zero-inflated-gp/onoffgpf/OnOffSVGP.py | endif
let s:l = 1 - ((0 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
1
normal! 0
lcd ~/Dropbox/Research/GPZIP
wincmd w
exe 'vert 1resize ' . ((&columns * 62 + 32) / 64)
exe 'vert 2resize ' . ((&columns * 1 + 32) / 64)
tabnext
edit ~/Dropbox/Research/GPZIP/gpzip3.py
set splitbelow splitright
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
argglobal
let s:l = 31 - ((30 * winheight(0) + 19) / 39)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
31
normal! 0
lcd ~/Dropbox/Research/GPZIP
tabnext
edit ~/Dropbox/Research/GPZIP/onoffgpf/OnOffLikelihood.py
set splitbelow splitright
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
argglobal
let s:l = 4 - ((3 * winheight(0) + 19) / 39)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
4
normal! 019|
lcd ~/Dropbox/Research/GPZIP
tabnext 1
set stal=1
badd +9 ~/Dropbox/Research/GPZIP/onoffgpf/OnOffSVGP.py
badd +39 ~/Dropbox/Research/GPZIP/gpzip3.py
badd +1 ~/Dropbox/Research/GPZIP/onoffgpf/OnOffLikelihood.py
badd +1 ~/Dropbox/Research/GPZIP/zero-inflated-gp/onoffgpf/OnOffSVGP.py
badd +31 ~/Dropbox/Research/GPZIP/zero-inflated-gp/onoffgpf/OnOffLikelihood.py
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 shortmess=filnxtToO
set winminheight=1 winminwidth=1
let s:sx = expand("<sfile>:p:r")."x.vim"
if file_readable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &so = s:so_save | let &siso = s:siso_save
nohlsearch
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
