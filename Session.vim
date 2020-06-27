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
edit onoffgpf/OnOffSVGP.py
set splitbelow splitright
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
argglobal
let s:l = 13 - ((0 * winheight(0) + 15) / 31)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
13
normal! 09|
lcd ~/Dropbox/Research/GPZIP
tabnext 1
badd +1 ~/.zhu_macos/zhuos.sh
badd +39 ~/Dropbox/Research/GPZIP/gpzip3.py
badd +0 ~/Dropbox/Research/GPZIP/onoffgpf/OnOffSVGP.py
badd +19 ~/Dropbox/Research/GPZIP/simGP.py
badd +22 ~/Dropbox/Research/GPZIP/svgp-regress.py
badd +1 ~/Dropbox/Research/GPZIP/regression_demo.py
badd +1 ~/Dropbox/Research/GPZIP/onoffgpf/OnOffLikelihood.py
badd +21 ~/Dropbox/Research/GPZIP/zero-inflated-gp/onoffgpf/OnOffSVGP.py
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
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
