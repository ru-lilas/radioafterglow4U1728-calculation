%
% notebook.cls
%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{notebook}[2026/02/09 Local class for research notebooks]

%=== Parent Class ===%
% We use ltjsarticle as parent class.
\LoadClass[
  12pt,
  a4paper,
  oneside
]{ltjsarticle}

%=== Packages ===%
\RequirePackage{import}
% \RequirePackage{titleps}
%--- math ---%
\RequirePackage{amsmath,amssymb,amsthm}
\RequirePackage{mathtools}
\RequirePackage{unicode-math}
\mathtoolsset{showonlyrefs}
\RequirePackage{derivative}
\RequirePackage{empheq}

%--- table and figure ---%
\RequirePackage{graphicx}
\RequirePackage{booktabs}
\RequirePackage{xcolor}
\RequirePackage{subcaption}

%--- physics ---%
\RequirePackage{physics2}
\usephysicsmodule{ab}
\RequirePackage[separate-uncertainty=true]{siunitx}

\RequirePackage{bxpapersize} %用紙サイズを一致させるパッケージ
\RequirePackage{enumitem}

%--- hyperref ---%
\RequirePackage[unicode=true]{hyperref}
\hypersetup{
  colorlinks=true,
  linkcolor=blue,
  citecolor=blue,
  urlcolor=blue,
  bookmarksopen=true,
}

%--- biblatex ---%
\RequirePackage[
  style=alphabetic,
  isbn=false,
  url=false,
  doi=false,
  eprint=false
]{biblatex}

%--- codeblock ---%
\RequirePackage{listings}

\RequirePackage{sty/subscripts}
\RequirePackage{sty/alias}

% =========================
% セクション見た目（軽め）
% =========================
\RequirePackage{titlesec}

\titleformat{\section}
  {\normalfont\Large\bfseries}
  {\thesection}{1em}{}

\titleformat{\subsection}
  {\normalfont\large\bfseries}
  {\thesubsection}{1em}{}

%=== 行間の設定 ===%
\RequirePackage{setspace}
\setstretch{1.15}

\AtBeginDocument{
  % ここに document 開始時に必ずしたい設定を書く
}
