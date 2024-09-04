
#import "@preview/physica:0.9.0": *
#import "@preview/i-figured:0.2.3"
#import "@preview/cetz:0.1.2" 
#import "@preview/xarrow:0.2.0": xarrow


#import cetz.plot 

#let title = "Elettronica"
#let author = "Bumma Giuseppe"

#set document(title: title, author: author)


#cetz.canvas({
  import cetz.draw: *
  // Your drawing code goes here
})

#show math.equation: i-figured.show-equation.with(level: 2, only-labeled: true)

#show link: set text(rgb("#cc0052"))

#show ref: set text(green)

#set page(margin: (y: 0.5cm))

#set heading(numbering: "1.1.1.1.1.1")
//#set math.equation(numbering: "(1)")

#set math.mat(gap: 1em)

//Code to have bigger fraction in inline math 
#let dfrac(x,y) = math.display(math.frac(x,y))

//Equation without numbering (obsolete)
#let nonum(eq) = math.equation(block: true, numbering: none, eq)
//Usage: #nonum($a^2 + b^2 = c^2$)

#let space = h(3.5em)
#let Space = h(5em)

//Shortcut for centered figure with image
#let cfigure(img, wth) = figure(image(img, width: wth))
//Usage: #cfigure("Images/Es_Rettilineo.png", 70%)

#let nfigure(img, wth) = figure(image("Images/"+img, width: wth))

//Code to have sublevel equation numbering
/*#set math.equation(numbering: (..nums) => {
   locate(loc => {
      "(" + str(counter(heading).at(loc).at(0)) + "." + str(nums.pos().first()) + ")"
    })
},)
#show heading: it => {
    if it.level == 1 {
      counter(math.equation).update(0)
    }
}*/
//

//Shortcut to write equation with tag aligned to right
#let tageq(eq,tag) = grid(columns: (1fr, 1fr, 1fr), column-gutter: 1fr, [], math.equation(block: true ,numbering: none)[$eq$], align(horizon)[$tag$])
// Usage: #tageq($x=y$, $j=1,...,n$)

// Show title and author
#v(3pt, weak: true)
#align(center, text(18pt, title))
#v(8.35mm, weak: true)

#align(center, text(15pt, author))
#v(8.35mm, weak: true)


#outline()



= SI

== Unità derivate SI

#table(
  columns: (13em, auto, auto, 13em),
  inset: 10pt,
  align: horizon,
  table.header(
    [*Grandezza*], [*Simbolo*], [*Unità SI non di base*], [*Unità SI di base*]
  ),
    [Carica elettrica],
    [$C$ (Coulomb)],
    [],
    [$s times A$],
    
    [Tensione elettrica e differenza di potenziale elettrico ],
    [$V$ "(Volt)"],
    [$ dfrac(W,A)$],
    [$m^2 times k g times s^(-3) times A^(-1)$],
    
    [Forza],
    [$N$ (Newton)],
    [],
    [$m times k g times s^(-2)$],

    [Energia/Lavoro],
    [$J$ (Joule)],
    [$N times m$],
    [$m^2 times k g times s^(-2)$],
    
    [Potenza],
    [$W$ (Watt)],
    [$dfrac(J,s)$],
    [$m^2 times k g times s^(-3)$],
     
    [Flusso magnetico],
    [$W b$ (Weber)],
    [$V times s$],
    [$m^2 times k g times s-2 times A^(-1)$],
     
    [Induzione magnetica],
    [$T$ (Tesla)],
    [$dfrac(W b,m^2)$],
    [$k g times s^(-2) times A^(-1)$],
     
    [Resistenza elettrica],
    [$Omega$ (Ohm)],
    [$dfrac(V,A)$],
    [$m^2 times k g times s^(-3) times A^(-2)$],
     
    [Conduttanza elettrica],
    [$S$ (Siemens)],
    [$dfrac(A,V)$],
    [$m^(-2) times k g^(-1) times s^3 times A^2$],
     
    [Capacità],
    [$F$ (Farad)],
    [$dfrac(C,V)$],
    [$m^(-2) times k g^(-1) times s^4 times A^2$],
     
    [Induttanza],
    [$H$ (Henry)],
    [$dfrac(W b,A)$],
    [$m^2 times k g times s^(-2) times A^(-2)$],
    
    [Frequenza],
    [$H z$ (Hertz)],
    [],
    [$s^(-1)$]
)

#pagebreak()

== Prefissi

#align(center)[
  #table(
    columns: (auto, auto, auto),
    inset: 10pt,
    align: horizon,
    table.header(
      [*Factor*], [*Name*], [*Symbol*]
    ),        
      [$10^(-24)$] , [yocto]  , [y],
      
      [$10^(-21)$] , [zepto] , [z],
      
      [$10^(-18)$] , [atto] , [a],
      
      [$10^(-15)$] , [femto] , [f],
      
      [$10^(-12)$] , [pico] , [p],
      
      [$10^(-9)$] , [nano] ,[n],
      
      [$10^(-6)$] , [micro] , [$mu$],
      
      [$10^(-3)$] , [milli] , [m],
      
      [$10^(-2)$] , [centi] , [c],
      
      [$10^(-1)$] , [deci] , [d],
      
      [$10^(1)$] , [deca] , [da],
      
      [$10^(2)$] , [hecto] , [mh],
      
      [$10^(6)$] , [mega] , [M],
      
      [$10^(9)$] , [giga] , [G],
      
      [$10^(12)$] , [tera] , [T],
      
      [$10^(15)$] , [peta] ,[P],
      
      [$10^(18)$] , [exa] , [E],
      
      [$10^(21)$] , [zetta] , [Z],
      
      [$10^(24)$] , [yotta] , [Y]
  )
]


#pagebreak()

= Tipi di esercizi

== D 
=== Formule notevoli
$
  C_(min) = "Cox" dot L_(min)^2 dot ("SP" + "SN") \
  "Resistenza equivalente pull-up" space R_("eq P") &= t_("LH")/(ln(2) dot C_(min)) \
  "Resistenza equivalente pull-down" space R_("eq N") &= t_("HL")/(ln(2) dot C_(min))\
  R_(P n) &= (R_("eq P") - dfrac(R_("RIF" P),S_P) dot N )/K \
  "Per percorsi critici" space R_P &= R_("eq P")/K\
  S_(P) &= R_("RIF" P)/(R P)
$
*Note:* 
- $ln(2) = 0,69$
- la $S_P$ che compare nella formula di $R_(P n)$ è sempre quella del percorso critico 
- $t_("LH")$ è il tempo di salita e $t_("HL")$ è il tempo di discesa. In generale negli esercizi se chiede di "dimensionare affinchè il tempo di salita al nodo $X$ sia inferiore o uguale a $Y p s$" vuol dire che prenderemo $t_("LH") = Y$.
- $p s$ sono pico secondi


=== Esame 14/06/2023
+ Della rete in figura si calcoli l'espressione booleana al nodo O.
+ Dimensionare i transistori pMOS affinchè il tempo di salita al nodo F sia inferiore o uguale a 90ps. Ottimizzare il progetto. Si tenga conto che i transistori dell'inverter di uscita hanno le seguenti geometrie : Sp = 200, Sn = 100.
+ Progettare la PDN
#cfigure("images/2024-08-02-17-44-29.png", 40%)

*Parametri tecnologici:*\
$
  &R_("RIF" P) = 10 k ohm space "si riferisce alla rete di pull-up" \ 
  &R_("RIF" N) = 5 k ohm space "si riferisce alla rete di pull-down"\ 
  &"Cox" = 7 f\F \/ mu m^2\
  &L_(min) = 0,25 mu m\
  &"Vdd" = 3V
$

*N.B.* I #text(fill: red)[numeri rossi] indicano la dimensione massima che possono assumere i transistor\

Per prima cosa si calcola $C_(min)$
$
  C_(min) &= "Cox" dot L_(min)^2 dot ("SP" + "SN") \
  &= 7 f\F \/ mu m^2 dot (0,25 mu m)^2 dot (200 + 100) \
  &= 131,35 f F
$
Poi la resistenza equivalente
$
  R_("eq P") = t_("LH")/(ln(2) dot C_(min)) &= (90 thin p s)/(0,69 dot 131,25 thin f F) \
  &=(90 dot 10^(-12) s)/(0,69 dot 131,25 dot 10^(-15))\
  &=0,99378 dot 10^3 thin ohm \
  &=993,79 thin ohm\
  &= 994 thin ohm
$

Per *dimensionare* si divide $R_("eq P")$ per il numero di transistor nel percorso critico.\
*Percorso critico:* percorso da $V_(c c)$ all'estremità in cui ci sono più transistor in serie (quando si considera il maggior numero di transistor in serie questi possono avere paralleli). Il percorso critico è anche il percorso con NMOS maggiore.

+ *Espressione booleana*
  
  *Regole:*
  - Gli elementi in serie sono il prodotto boolenano degli elementi
  - Gli elementi in parallelo sono la somma booleana deli elementi

  PD := rete di pull-down \
  PU := rete di pull-up

  Rete di pull-up al nodo $F$:
  $
    P U = ((C dot B) + overline(A)) dot C + A dot overline(C) = F
  $
  La rete di pull-down si calcola invertendo somma e prodotto e negando poi tutta l'espressione
  Scriviamo $F$ in forma negata
  $
    F &= overline( (((C + B) dot overline(A)) + C) dot (A + overline(C)))
  $
  allora
  $
    O = overline(F) &= overline( overline( (((C + B) dot overline(A)) + C) dot (A + overline(C))) ) \
    &= overline( ((( overline(C) dot overline(B)) +A) dot overline(C)) + (overline(A) dot C))
  $

+ *Dimensionare i transistor*
  
  *Primo caso peggiore*

  Si calcola la $R P$, che solo per il percorso critico vale
  $(R_("eq P"))/("nMOS")$. In questo caso il percorso critico è $X B C$; la $X$ sta a significare che il valore di $A$ non ci interessa.

  $
    R_(P) &= (994 thin ohm)/3\
    &=331,33 thin ohm \
    &= 331 thin ohm
  $
  Quindi ora calcoliamo la $S P$ con la formula
  $
    S_P = (R_("RIF P"))/(R_P) &= (10 thin k ohm)/(331 thin ohm)\
    &= 30,21\
    &= 31
  $
  *N.B.* Arrotondare sempre all'intero successivo

  *Secondo caso peggiore*

  Per ottimizzare un percorso non critico si ha una formula che varia in base alle caratteristiche del percorso stesso
  $
    R_P = (R_("eq P") - dfrac(R_("RIF" P),S P) dot N )/K 
  $
  dove $N$ è il numero di MOS del percorso critico che interessano anche un percorso non critico e $K$ è il numero di MOS del percorso non critico cosiddetti "nuovi", cioè che non fanno parte del percorso critico. Inoltre $K+N$ è il numero di MOS del percorso non critico; quando si devono calcolare $K$ e $N$ di solito si calcola prima $K$ e poi si ricava $N$ dall'ultima formula.

  In questo caso consideriamo $A X overline(C)$. Abbiamo 2 pMOS nuovi e nessun pMOS del percorso critico, quindi $N=0$ e $K=2$
  $
    R_(P 2) = (R_("eq P") - cancel(dfrac(R_("RIF" P),S P) dot overbrace(N, 0)) )/K &= 994/2 thin ohm\
    &= 497 thin ohm
  $
  $
    S P_2 = (R_("RIF P"))/(R_(P 2)) &= (10000 thin ohm)/(497 thin ohm)\
    &=20,12\
    &=21
  $

  *Terzo caso*

  Consideriamo il percorso $overline(A) thin overline(B) C$. Abbiamo un nMOS nuovo e un nMOS del percorso critico, quindi $N=1$ e $K=1$.\
  *N.B.* Bisogna specificare $overline(B)$ e non $X$ perché si deve cosiderare solo il percorso di $overline(A) C$, e se $B$ fosse accesso il percorso sarebbe diverso.

  $
    R_(P 3) = (R_("eq P") - dfrac(R_("RIF" P),S P) dot N)/K &= (994 thin ohm - dfrac(10000 thin ohm,31) dot 1)/1\
    &=994 thin ohm - 323 thin ohm \
    &= 671 thin ohm
  $
  $
    S P_2 = (R_("RIF P"))/(R_(P 2)) &= (10000 thin ohm)/(671 thin ohm)\
    &=14,9\
    &=15
  $

  *Nota:* le $S_P$ trovate denotano la dimensione massima dei transistori che interessano il percorso; in particolare si assegna prima la dimnesione ai transistori presenti nel percorso critico, poi agli altri, in modo che il valore trovato per un transistori del percorso critico sia dominante rispetto al valore trovato per lo stesso transistore per un percorso non critico.

+ *Progettare la PDN*
  
  La formula della rete di pull-down è la seguente (prima l'abbiamo calcolata scrivendola in forma negata)
  $
    P D = (((C+B) dot overline(A)) + C) dot (A+overline(C))
  $
  quindi, seguendo le regole dell'algebra booleana, la rete può essere rappresentata come segue
  #cfigure("images/Screenshot_20240806-182202.png",40%)



=== Esame 12/06/2024
+ Determinare l'espressione booleana al nodo O
+ Dimensionare i transistori nMOS e pMOS in modo che i tempi di salita e discesa, al nodo F, siano inferiori o uguali a $100 thin p s$. Si ottimizzi il progetto per minimizzare l'area occupata da tutti i transistori.
Si tenga conto che i transistori dell'inverter di uscita hanno le seguenti geometrie : $S_P=300$, $S_n= 150$.
  
*Parametri tecnologici:*
$
  R_("rif" p) &= 10 thin k ohm \
  R_("rif" n) &= 5 thin k ohm \
  C_(o x) &= 7 thin f F \/mu m^2 \
  L_(min) &= 0,25 thin mu m \
  V_(C C) &= 3,3 V
$

#cfigure("images/2024-08-07-16-26-24.png", 45%)

+ *Espressione booleana*
  $
    "PD" = ( (C+B) dot (A+D) dot (B+ overline(C)) ) dot "CLK" + overline("CLK")
  $
  Nota: Il CLK non negato è in serie con il resto del circuito della rete di pull-down, quello negato è in parallelo a tutta la rete di pull-down.

  $
    "F" = overline("PD")\
    "O" = overline("F") = overline(overline("PD")) \
    "O" = overline(overline("PD"))
  $
  quindi
  $
    "O" &= overline(overline(( (C+B) dot (A+D) dot (B+ overline(C)) ) dot "CLK" + overline("CLK"))) \
    &= overline(( (overline(C) dot overline(B)) + (overline(A) dot overline(D)) + (overline(B) dot C) + overline("CLK") ) dot "CLK" )
  $
+ *Dimensionare transistori nMOS e pMOS*
  
  - Rete pull-up\
    C'è solo un CLK nella rete di pull-up, quindi $K=1$
    $
      space space R_P  = dfrac(R_("eq" P), K) = 736 thin ohm space
      S_P = dfrac(R_("RIF" P), R_P) &= dfrac(10000 thin ohm, 736 thin ohm)\
      &= 13,58\
      &= 14
    $
  - Rete pull-down
    In questo caso ci sono diversi percorsi critici:
    - $A B C overline(D)$
    - $overline(A) B C D$
    - $A B overline(C) overline(D)$
    - $overline(A) B overline(C) D$

    Il numeri di MOS in un percorso non è sempre un intero, infatti, se ci sono dei transistor in parallelo, il numero di MOS corrispondente è uguale a $dfrac(1,"numero di transistor in parallelo")$

    #cfigure("images/Screenshot_20240807-163312.png", 70%)

    Quindi per tutti i percrosi critici individuati $K=3,5$

    $
      R_N = dfrac(R_("eq" N),K) &= dfrac(736 thin ohm,3.5) &space S_N = dfrac(R_("RIF" N), R_N) &= dfrac(5000 thin ohm,210 thin ohm) \
      &= 210 thin ohm &space &=24
    $

    Tutti i transistori della rete pull-down avranno quindi dimensione 24

    #cfigure("images/2024-08-07-18-12-47.png", 50%)





































