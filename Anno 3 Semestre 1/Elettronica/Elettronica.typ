
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





= Tipi di esercizi

== D 
+ Della rete in figura si calcoli l'espressione booleana al nodo O.
+ Dimensionare i transistori pMOS affinchè il tempo di salita al nodo F sia inferiore o uguale a 90ps. Ottimizzare il progetto. Si tenga conto che i transistori dell'inverter di uscita hanno le seguenti geometrie : Sp = 200, Sn = 100.
+ Progettare la PDN
#cfigure("images/2024-08-02-17-44-29.png", 40%)

*Parametri tecnologici:*\
$R_("RIF" P) = 10 k ohm$\ 
$R_("RIF" N) = 5 k ohm$\
$"Cox" = 7 f\F \/ mu m^2$\
$L_(min) = 0,25 mu m$\
$"Vdd" = 3V$

*N.B.* I #text(fill: red)[numeri rossi] indicano la dimensione massima che possono assumere i transistor\

=== Formule notevoli
$
  C_(min) = "Cox" dot L_(min) dot ("SP" + "SN") \
  "Resistenza equivalente pull-up" space R_("eq P") &= t_("LH")/(ln(2) dot C_(min)) \
  "Resistenza equivalente pull-down" space R_("eq N") &= t_("HL")/(ln(2) dot C_(min))
$
*N.B.* $ln(2) = 0,69$

Con $t_("LH")$ tempo di salita e $t_("HL")$ tempo di discesa. In generale negli esercizi se chiede di "dimensionare affinchè il tempo di salita al nodo $X$ sia inferiore o uguale a $Y$ ps" vuol dire che prenderemo $t_("LH") = Y$.

*N.B.* ps sono pico secondi

Per prima cosa si calcola $C_(min)$
$
  C_(min) &= "Cox" dot L_(min) dot ("SP" + "SN") \
  &= 7 f\F \/ mu m^2 dot (0,25 mu m)^2 dot (200 + 100) \
  &= 131,35 f F
$
Poi la resistenza equivalente
$
  R_("eq P") = t_("LH")/(ln(2) dot C_(min)) &= (90 thin p s)/(0,69 dot 131,25 thin f F) \
  &=(90 dot 10^(-9) s)/(0,69 dot 131,25 dot 10^(-12))\
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

  Reti di pull-up al nodo $F$:
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
  $(R_("eq P"))/("nMOS")$. In questo caso il percorso critico è $X B C$; la $X$ sta a significare che il valore di $A$ non ci interessa; se un elemento è negato vuol dire che il transistor è acceso.

  $
    R_(P) &= (994 thin ohm)/3\
    &=331,33 thin ohm \
    &= 331 thin ohm
  $
  Quindi ora calcoliamo la $S P$ con la formula
  $
    S P = (R_("RIF P"))/(R P) &= (10 thin k ohm)/(331 thin ohm)\
    &= 30,21\
    &= 31
  $
  *N.B.* Arrotondare sempre all'intero successivo

  *Secondo caso peggiore*

  Per ottimizzare un percorso non critico si ha una formula che varia in base alle caratteristiche del percorso stesso
  $
    R_P = (R_("eq P") - dfrac(R_("RIF" P),S P) dot N )/K 
  $
  dove $N$ è il numero di MOS del percorso critico che interessano anche un percorso non critico e $K$ è il numero di MOS del percorso non critico cosiddetti "nuovi", cioè che non fanno parte del percorso critico.

  In questo caso consideriamo $A X overline(C)$. Abbiamo 2 pMOS nuovi e nessun pMOS del percorso critico, quindi $N=0$ e $K=2$
  $
    R_(P 2) = (R_("eq P") - cancel(dfrac(R_("RIF" P),S P) dot overbrace(N, 0)) )/K &= 994/2\
    &= 497
  $














































