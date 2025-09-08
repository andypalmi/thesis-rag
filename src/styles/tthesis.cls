\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{styles/tthesis}[2021/06/21 Example LaTeX class]

\LoadClass[a4paper,english,twoside,openright,11pt]{book}

% makes PDF searchable
\usepackage{cmap}

\usepackage[english]{babel}

%\usepackage[latin9]{inputenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern} % Latin Modern font
\usepackage[T1]{fontenc}

\usepackage{csquotes}

\usepackage{textcomp} % needed for fontenc
%\usepackage{styles/extref}
\usepackage{xurl}
\usepackage[final]{listings}
	%\lstset{numbers=left,breaklines}
\usepackage{acronym}
\usepackage{makeidx}
	\makeindex
\usepackage[binding=0.8cm]{layaureo} % better handling of A4 format
\usepackage{tocbibind} 
\usepackage[labelfont=bf]{caption}[2004/07/16] % format of captions
\usepackage{subfig}

% tables
\usepackage{booktabs}
\usepackage{tabulary}
\usepackage{lscape,longtable,booktabs}
\usepackage{geometry}
\usepackage{array}
\usepackage{changepage}
\usepackage{multirow}
\usepackage{rotating}

% Algorithm
\usepackage{algorithm}
\usepackage{algpseudocode}

% citations at the beginning of each chapter
\usepackage{epigraph}
	\setlength{\epigraphwidth}{0.59\textwidth}
	\renewcommand{\epigraphsize}{\small}
	\renewcommand{\epigraphflush}{flushleft}

\usepackage{styles/reallyblank} % insert empty page at the beginning of a chapter with empty headings

\usepackage{tcolorbox}
\tcbuselibrary{skins, listings,breakable}
\tcbset{
  promptbox/.style={
    enhanced,
    boxsep=4pt, left=4pt, right=4pt,
    colback=gray!10, colframe=gray!50,
    left=1mm, right=1mm, top=1mm, bottom=1mm,
    before skip=1ex, after skip=1ex,
    sharp corners,
    fonttitle=\bfseries,
    fontupper=\ttfamily\small,
    title={#1}
  }
}

% the following is for thesis in Italian language; adapt it for English ones
\usepackage{sistyle}
\AddToSIstyle{English}{
   \SIdecimalsign{.}
   \SIthousandsep{\,}
   \SIproductsign{\cdot}
   \SIunitsep{\,}
   \SIunitspace{\,}
   \SIunitdot{\cdot}
   \SIobeyboldfalse
   \SIgroupfourfalse}
\SIstyle{English}

% used to use the footnote inside a figure's caption
\usepackage{afterpage}

\usepackage{gensymb}
\usepackage{wasysym}
\usepackage{amsfonts}
\usepackage{graphicx}

\usepackage{verbatim}
\usepackage{courier}
\usepackage{float}

\lstset{numbersep=5pt,
numberstyle=\tiny,
numbers=none,
frame=lines,
basicstyle=\linespread{1.0}\ttfamily\footnotesize
}

\usepackage{cleveref}

% Abstract Environment Definition
\newenvironment{abstract}{%
  \cleardoublepage % Ensure abstract starts on a new page
  \addcontentsline{toc}{chapter}{Abstract}% % Add to Table of Contents
  \thispagestyle{empty}% % No page number on this page
  \begin{center}%
    \bfseries\Huge Abstract% % Single, prominent "Abstract" title
  \end{center}%
  \vspace{1em}% % Space after the title
  \quotation% % Indent the abstract text
}{%
  \endquotation%
}

\newcommand{\printtitle}{\@title}
\newcommand{\printauthor}{\@author}
\newcommand{\regnumber}{N. Matricola}

% Set linespread to 1 in the verbatim environment
\makeatletter
\def\verbatim@font{\linespread{1}\normalfont\ttfamily}
\makeatother

% Set default linespread for the whole document
\renewcommand{\baselinestretch}{1.3}

%-------------------------------------------------------------------
% Frontpage
%-------------------------------------------------------------------

\makeatletter
\newcommand{\graduation}[1]{\def\@graduation{#1}}
\newcommand{\printgraduation}{\@ifundefined{@graduation}{}{\@graduation}}
\newcommand{\academicyear}[1]{\def\@academicyear{#1}}
\newcommand{\printacademicyear}{\@ifundefined{@academicyear}{}{\@academicyear}}
\newcommand{\supervisor}[1]{\def\@supervisor{#1}}
\newcommand{\printsupervisor}{\@ifundefined{@supervisor}{}{\@supervisor}}
\newcommand{\cosupervisor}[1]{\def\@cosupervisor{#1}}
\newcommand{\printcosupervisor}{\@ifundefined{@cosupervisor}{}{\@cosupervisor}}
\makeatother

\newcommand{\frontpagefile}{frontpages/frontpage-unipv.inc}

\newcommand{\printfrontpage}{%
\input{\frontpagefile}%
\thispagestyle{empty}%
\mbox{}%
\newpage%
}
