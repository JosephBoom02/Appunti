
#import "@preview/physica:0.9.0": *
#import "@preview/i-figured:0.2.3"
#import "@preview/cetz:0.1.2" 
#import "@preview/xarrow:0.2.0": xarrow


#import cetz.plot 

#let title = "Elettronica"
#let author = "Bumma Giuseppe"

#set page(numbering: "1")

#set document(title: title, author: author)


#cetz.canvas({
  import cetz.draw: *
  // Your drawing code goes here
})

#show math.equation: i-figured.show-equation.with(level: 2, only-labeled: true)

#show link: set text(rgb("#cc0052"))

#show ref: set text(green)

#set page(margin: (y: 1cm))

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
//Usage: #cfigure("images/Es_Rettilineo.png", 70%)

#let nfigure(img, wth) = figure(image("images/"+img, width: wth))

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

// Color in math mode
#let colpurp(x) = text(fill: purple, $#x$)
#let colred(x) = text(fill: red, $#x$)


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


== A


=== Formule notevoli
*N.B.* Alcune formule si trovano nel #link("./Formulario_elettronica.pdf")[Formulario].


==== Formule simboliche di resistore, induttore e condensatore
$
  &bold("Resistore") &space space &bold("Induttore") &space space &bold("Condensatore") 
  \
  &V =  R I   &   &V = j omega L I  &   &V = - dfrac(j, omega C) I
  \
  &Z = R      &   &Z = j omega L    &   Z = &dfrac(1, j omega C) = - dfrac(j,omega C)
$


=== Diagrammi di Bode


==== Guadagno statico
$
    G_a (j omega) &= mu 
    & space 
    |G_a (j omega)|_("dB") &= 20 log|mu| 
    & space   
    arg(G(j omega)) =  arg( mu)
$

#cfigure("images/Diagramma_guadagno_statico_1.png", 63%)

Per quanto riguarda il diagramma dell'ampiezza,
- se $|mu|  >= 1$ allora $20 log|mu|   >= 0$
- se $|mu| <1$ allora $20 log|mu|  < 0$.  
Per il diagramma della fase invece
- se $ mu >0$ allora $ arg( mu)=0$
- se $ mu <0$ allora $ arg( mu)=-180 degree$.


==== Zeri nell'origine
Consideriamo una risposta con uno zero nell'origine (cioè $g=-1$)
$
    G_b (j omega) &=  frac(1,(j omega)^(-1)) = j omega 
    &wide 
    |G_b (j omega)|_("dB") &= 20  log  omega 
    &wide  
    arg(G_b (j  omega)) &=  arg(j omega)
$

#cfigure("images/Diagramma_zero_origine.png", 72%)

La retta che definisce l'ampiezza $ log  omega arrow.r.bar 20  log  omega$ ha pendenza $20 "dB/dec"$; se ho $g$ zeri nell'origine allora la pendenza della retta sarà $20 dot g "dB/dec"$.\
$j omega$ è un punto sul semiasse immaginario positivo $forall omega > 0$, quindi fase $90^degree forall > 0$.


==== Poli nell'origine
Consideriamo una risposta con un polo nell'origine (cioè $g=1$)
$
    G_b (j omega) &=  frac(1,(j omega)^1) =  frac(1,j omega) 
    &space  
    |G_b (j omega)|_("dB") &= -20  log  omega 
    &space  
    arg(G_b (j  omega)) &= - arg(j omega)
$

#cfigure("images/Diagramma_poli_origine.png", 75%)

Anche in questo caso, se ho $g$ poli nell'origine allora la pendenza della retta sarà $-20  "dB/dec"$.  
Per quanto riguarda la fase $-j  omega$ è un punto sul semiasse immaginario negativo $ forall  omega>0$, quindi la fase è $-90^ degree$.



==== Zero reale (ampiezza)
Consideriamo una risposta con uno zero reale
$
    G_c (j omega) = 1 + j  omega  tau  
$
$
    |G_c (j omega)|_("dB") = 20  log  sqrt(1 +  omega^2 tau^2)  approx
    cases(
        20  log 1 = 0 & omega  <<  dfrac(1,|tau|)\  
        20  log  omega |tau| = -20  log  dfrac(1,|tau|) + 20  log  omega wide& omega  >>  dfrac(1,|tau|)
    )
$

$
    lr(|G_c (j  frac(1,|tau|)  )|)_("dB") &= 20  log  sqrt(1+ frac(1,|tau|^2) tau^2) \
    &= 20  log  sqrt(2)  approx 3
$
per $ omega =  frac(1,|tau|)$ abbiamo lo scostamento massimo.

#nfigure("Zero_reale_ampiezza.png", 60%)

==== Zero reale negativo (fase)
Consideriamo una una risposta con uno zero reale negativo
#tageq($G_c (j omega) = 1 + j  omega  tau$, $ tau > 0$)

$
    arg(G_c (j omega)) =  arg(1+j omega  tau)  approx 
    cases(
        0 wide& omega  <<  frac(1, tau) \ 
        90^ degree &  omega  >>  frac(1, tau)
    )    
$
Se $ omega  arrow 0$ allora $ arg(1+j omega tau)  approx  arg(1)=0$

#cfigure("images/Diagramma_zero_reale_negativo_1.png", 60%)

se $ omega  >>  dfrac(1, tau)$ allora graficamente

#cfigure("images/Diagramma_zero_reale_negativo_2.png", 60%)
#cfigure("images/Diagramma_zero_reale_negativo_3.png", 72%)

il cambio di fase inizia circa una decade prima e finisce circa una decade dopo la pulsazione di taglio $ omega =  dfrac(1, tau)$.

#cfigure("images/Diagramma_zero_reale_negativo_4.png", 60%)

$ dfrac(1,5) dot dfrac(1, tau) = 0.2 dot dfrac(1, tau) = 2  dot 10^(-1)  dfrac(1, tau)$, il doppio in scala logaritmica è $ dfrac(1,3)$ di una decade.


==== Zero reale positivo (fase)
Consideriamo $G_c (j omega) = 1 + j omega  tau,    tau <0$ (cioè una risposta con uno zero reale positivo)

#cfigure("images/Diagramma_zero_reale_positivo_1.png", 32%)

$
     arg(G(j omega)) =  arg(1+j omega tau)  approx 
    cases(
        0 &  omega  <<  frac(1,|tau|) \  
        -90^( degree) wide&  omega  >>  frac(1,|tau|)
    )   
$

#cfigure("images/Diagramma_zero_reale_positivo_2.png", 70%)



==== Polo reale
Consideriamo $G_c (j omega) =  dfrac(1,1+j omega T)$ (cioè una risposta con un polo reale)
$
    |G_c (j omega)|_("dB") &= 20  log  lr(| frac(1,1+j omega  T) |)  
    \
    &= -20  log |1+j omega T|
$
$
    |G_c (j omega)|_("dB") &= -20  log  sqrt(1+ omega^2T^2) 
    & space  
    arg(G_c (j omega)) = - arg(1+j omega T)
$

#cfigure("images/Diagramma_polo_reale.png", 73%)

Il diagramma è uguale al diagramma dello zero ma ribaltato rispetto all'asse reale (consistentemente con il segno di $T$).



==== Polo reale negativo
Consideriamo $G_c (j omega) =  dfrac(1,1+j omega T),   T>0$ (cioè una risposta con un polo reale negativo)
$
    |G_c (j omega)|_("dB") &= -20  log  sqrt(1+ omega^2T^2) 
    & space  
    arg(G_c (j omega)) = - arg(1+j omega T)
$

#cfigure("images/Diagramma_polo_reale_negativo_1.png", 53%)

Fino a $ dfrac(1,T)$, (pulsazione di taglio), si ha un andamento costante a $0 "dB" $, cioè il modulo della sinusoide in uscita non cambia. A partire da $ dfrac(1,T)$ si ha una retta $ log omega  arrow.r.bar 20  log  dfrac(1,|T|) - 20  log  omega$ con pendenza $-20   "dB/dec"$.  
    
Lo scostamento massimo (tra diagramma asintotico e diagramma reale) si ha in $ omega =  dfrac(1,T)$ dove
$
    |G_c (j omega)|_("dB") &= -20  log  sqrt(1+1) \ 
    &= -20  log  sqrt(2)  approx -3
$
Il cambio di fase inizia circa una decade prima e finisce circa una
decade dopo la pulsazione di taglio $ omega =  dfrac(1,T)$. 



==== Zeri complessi coniugati (ampiezza)
Consideriamo $G_d (j omega) = 1+ 2j  zeta  dfrac( omega, alpha_n) -  dfrac( omega^2, alpha_n^2)$, una risposta con una coppia di zeri complessi coniugati
$
    |G_d (j omega)|_("dB") = 20  log  sqrt( (1 -  frac( omega^2, alpha_n^2) )^2 + 4  zeta^2  frac( omega^2, alpha_n^2))
$
per $ omega  >>  alpha_n$
$
    |G_d (j omega)|_("dB") & approx 20  log  sqrt( ( frac( omega^2, alpha_n^2) )^2)  \
    &=20  log  frac( omega^2, alpha_n^2)  \
    &= 20  log  ( frac( omega, alpha_n) )^2  \
    &= 40  log  frac( omega, alpha_n)  \
    &=  underbrace(40  log  omega, "variabile") -  underbrace(40  log  alpha_n, "costante")
$
Quindi la risposta si comporta come una retta, di pendenza pari a 40 dB.  
Analizziamo ora la risposta per $ omega =  alpha_n$
$
    |G_d (j omega)|_("dB") &20  log  sqrt( (1 -  frac( omega^2, alpha_n^2) )^2 + 4  zeta^2  frac( omega^2, alpha_n^2))  \
    &= 20  log  sqrt(4  zeta^2)  \
    &= 23  log 2|zeta|  \
    &=  underbrace(20 log 2 ,6   "dB") +20  log |zeta|
$
quindi scostamento significativo dipendente dal valore di $ zeta$.

$
    |G_d (j omega)|_("dB") = 20  log  sqrt( (1 -  frac( omega^2, alpha_n^2) )^2 + 4  zeta^2  frac( omega^2, alpha_n^2))  approx
    cases(
        20  log(1) = 0 &  omega  <<  alpha_n  \
        20  log  dfrac( omega^2, alpha_n^2) = -40  log  alpha_n + 40  log  omega wide& omega   >>  alpha_n
    )
$

#cfigure("images/Diagramma_zeri_cc_ampiezza_1.png", 55%)
#cfigure("images/Diagramma_zeri_cc_ampiezza_2.png", 55%)

Il minimo dell'ampiezza si ha alla pulsazione $ omega_r =  alpha_n  sqrt(1-2 zeta^2)$ con $|G_d (j omega_r)| = 2 |zeta| sqrt(1- zeta^2)$


==== Zeri complessi coniugati a parte reale negativa (fase)
Consideriamo $G_d (j omega) = 1+ 2j  zeta  dfrac( omega, alpha_n) -  dfrac( omega^2, alpha_n^2),    zeta>0$, una risposta con una coppia di zeri complessi coniugati a parte reale negativa
$
     arg(G_d (j  omega))  approx 
    cases(
        0 &  omega  <<  alpha_n  \
        180^ degree wide& omega  >>  alpha_n
    )    
$

#cfigure("images/Diagramma_zeri_cc_neg_1.png", 60%)

Vediamo che la risposta, per $ omega  >>  alpha_n$, nel piano complesso ha sicuramente una parte reale molto negativa, mentre la parte immaginaria dipende dal valore $ zeta$, il quale influenza molto l'andamento della fase. Ad esempio se $ zeta  arrow 0$ la parte immaginaria nel piano complesso tenderà anch'essa a 0, così da rendere molto facile appurare che l'argomento della nostra risposta sia quasi $180^ degree$. Anche con $ zeta =1$ l'argomento sarà circa $180^ degree$, ma solo per pulsazioni molto grandi, perché la parte reale tende a $ dfrac( omega^2, alpha_n^2)$, che è $O( dfrac( omega, alpha_n))$ (la parte immaginaria) per $ omega  arrow  infinity$.

#cfigure("images/Diagramma_zeri_cc_neg_2.png", 73%)

Nel diagramma di fase più $ zeta$ è piccolo e più la discontinuità da $0^ degree$ a $180^ degree$ è rapida. 



==== Zeri complessi coniugati a parte reale positiva
Consideriamo $G_d (j omega) = 1+ 2j  zeta  dfrac( omega, alpha_n) -  dfrac( omega^2, alpha_n^2),  quad  zeta < 0$, una risposta con una coppia di zeri complessi coniugati a parte reale positiva.  
Il diagramma di fase di è speculare a quello precedente

#cfigure("images/Diagramma_zeri_cc_pos_1.png", 73%)




==== Poli complessi coniugati a parte reale negativa <poli_complessi_coniugati_parte_reale_negativa>
Consideriamo una risposta in frequenza con poli complessi coniugati a parte reale negativa
$G_d (j omega) =  dfrac(1,1+2j  xi  frac( omega, omega_n)- frac( omega^2, omega_n^2)),    xi > 0$

#cfigure("images/Diagramma_poli_cc_neg_1.png", 63%)

I diagrammi sono quelli precedenti ribaltati rispetto all'asse reale, infatti la retta del diagramma di ampiezza asintotico dopo la pulsazione $ omega_n$ ha pendenza $-40$ dB/dec.  
Il picco di risonanza si trova alla pulsazione (di risonanza) $ omega_r =  omega_n  sqrt(1-2 xi^2)$ con $|G_d (j omega_r)| =  dfrac(1,2|xi| sqrt(1-2 xi^2))$; alla frequenza $ omega_n$ si ha $|G_d (j omega_n)| =  dfrac(1,2|xi|)$
   
Soffermiamoci un attimo sul caso in cui $ xi  arrow 0$: se do una sinusoide con frequenza inferiore a $ omega_n$ essa non viene sfasata; se invece la sua frequenza è di poco superiore a $ omega_n$ la sua fase viene sfasata di $90^ degree$; il modulo viene amplificato di molto se la frequenza della sinusoide è nell'intorno di $ omega_n$.



==== Poli complessi coniugati a parte reale positiva
Consideriamo una risposta in frequenza con una coppia di poli complessi coniugati a parte reale positiva

#tageq($G_d (j omega) =  frac(1,1+2j  xi  frac( omega, omega_n) -  frac( omega^2, omega_n^2))$, $ xi < 0$)

Calcoliamo i poli
$
    G_d (s) =&  frac(1,1+2  frac( xi, omega_n)s +  frac(s^2, omega_n^2))  
    \
    =&  frac( omega_n^2,s^2+2  xi  omega_n s +  omega_n^2)  
    \
    ==>& p_(1 slash 2) = -  xi  omega_n  plus.minus j  omega_n  sqrt(1 -  xi^2)
$

#cfigure("images/Diagramma_poli_cc_pos.png", 68%)

Diagramma ottenuto da quello degli zeri (caso $ zeta<0$) ribaltando rispetto all'asse reale.










=== Esame 14/09/2021
#cfigure("images/Esame_14-09-2021.jpg",43%)
#cfigure("images/Screenshot_20240904-100842.png", 55%)

#set enum(numbering: "1)")

+ Usiamo la sovrapposizione degli effetti per calcolare $v_O$
  #circle(radius: 7pt)[#set align(center + horizon) 
                        1] $v_a != 0, v_B = 0 $\
   \
   $
    space space underbrace(v_(O_1) &= dfrac(R_2,R_1) dot v_a, "amplificatore lineare " \ "invertente")
    space space
    underbrace(v_("OUT"_A) &= - dfrac(R_4,R_3) dot v_(O_1) , "amplificatore lineare" \ "invertente")
    =  dfrac(R_4,R_3) dot  dfrac(R_2,R_1) dot v_A
   $

   #circle(radius: 7pt)[#set align(center + horizon) 
                        2] $v_a = 0, v_B != 0 $\
    $
      space space v_(O_1) = 0 space space wide underbrace(v_("OUT"_B) = (1 +  dfrac(R_4,R_3) dot v_B), "amplificatore lineare" \ "non invertente") =  dfrac(R_3 + R_4,R_3) dot v_B
    $
  
  
  $
    v_"OUT" = v_("OUT"_A) + v_("OUT"_B) =  colpurp( dfrac(R_4,R_3) dot  dfrac(R_2,R_1) ) dot v_A + colpurp( dfrac(R_3 + R_4,R_3) ) dot v_B
  $
+ #text(fill: purple, size: 12pt)[Sommatore tra $v_A$ e $v_B$]\
  Si eguagliano i coefficienti davanti a $v_A$ e $v_B$

  $
    colpurp( dfrac(R_4,R_3) dot  dfrac(R_2,R_1) ) = colpurp( dfrac(R_3 + R_4,R_3) ) &==> 
    dfrac(40 k ohm dot 40 k ohm, 10 k ohm dot R_1) = (10 k ohm + 40 k ohm, 10 k ohm ) \
    &==> dfrac(160 k ohm, R_1) = 5 k ohm \
    &==> R_1 = dfrac(150,5) k ohm = 32 k ohm
  $
  


=== Esame 28/01/2021
#cfigure("images/Esame_28-01-2021.jpg", 45%)
#cfigure("images/Screenshot_20240904-113654.png", 35%)


+ Usiamo la sovrapposizione degli effetti per calcolare $v_0$
  #circle(radius: 7pt)[#set align(center + horizon) 
                        1] $v_a != 0, v_B = 0 $, abbiamo come $"OPAMP"_1$ un FOLLOWER e come $"OPAMP"_2$ un NON INVERTENTE \
   
   #cfigure("images/ResizedImage_2024-09-04_11-47-37_2619.png",45%)

   $
    space space underbrace(v_x &= 0, "follower spento")
    space space
    v_y &= (1 + dfrac(20 R,10 R) ) dot v_A \
    &=  3 v_A
   $

   $
      i_(0_1) = dfrac(3v_A,R)
   $

   #circle(radius: 7pt)[#set align(center + horizon) 
                        2] $v_a = 0, v_B != 0 $, abbiamo come $"OPAMP"_1$ un FOLLOWER e come $"OPAMP"_2$ un INVERTENTE \ 

    #cfigure("images/ResizedImage_2024-09-04_11-49-06_2358.png",45%)

    $
      space space underbrace(v_x = v_B, "follower acceso") 
      space space wide 
      v_y = - dfrac(20 R,10 R) dot v_x &=  dfrac(R_3 + R_4,R_3) dot v_B \
      &= - dfrac(20 R,10 R) dot v_B \
      &= -2 v_B
    $
  
  
  $
    v_y - v_x - v_(0_2) = 0 ==> v_(0_2) = v_y - v_x
  $

  $
    i_(0_2) &= dfrac(v_y - v_B, R)\
    &=dfrac(-2 v_B - v_B, R)\
    &= - dfrac(3v_b,R)
  $

  $
    i_0 = i_(0_1) + i_(0_2) &= dfrac(3v_A,R) - dfrac(3v_b,R)\
    &= dfrac(3v_A - 3v_b,R)\
    &= dfrac(3,R) (v_A - v_B)
  $

+ #text(fill: purple, size: 12pt)[Modulo massimo corrente]\
  $
    i_("OUT"_"MAX") = i_0 + i_1 &=  dfrac(3,R) (v_A - v_B) + dfrac(3v_a - 2v_B - v_A,20R)\
    &= dfrac(1,R) ( dfrac(30v_A - 30 v_B +v_A - v_B,10) )\
    &= dfrac(31,10R) (v_A - v_B)
  $
  Adesso, siccome il testo dell'esercizio ci dice che $v_A$ e $v_B$ possono assumere valori all'interno dell'intervallo $[-2V .. thin 2V]$, impostiamo i seguenti valori: $v_A = 2V$, $v_B = -2V$

  $
    i_("OUT"_"MAX") = dfrac(31,10R) dot 4 V\
    &= dfrac(124 V,10 R)\
    &=1,24 dot 10^(-4) A \
    &= 12,4 thin m A
  $





=== Esame 08/02/2021 (Diagramma di Bode standard + saturazione)

#cfigure("images/Esame_08-02-2021_Disegno_1.png",40%)

Dati:
$
  R_1 &= 10 k ohm &space space R_2 &= 90 k ohm
  \
  C &= 180 p F & L_+=L_- &= 10V
$

#text(fill: purple, size: 12pt)[1) Calcolo funzione di trasferimento]\

È stata ricavata la seguente funzione di trasferimento
$
  H(j omega) = dfrac( 1+ dfrac( j omega C R_2, 10 ),1 + j omega C R_2 )
$
Calcoliamo i valori della funzione di trasferimento a $0$ e a $+ infinity$
$
  omega -> 0 ==> H(j omega) = 10 space space omega -> + infinity ==> H(j omega) = 1
$
e trasformiamo tali valori in dB
$
  abs(10)_"dB" = 20 "dB" space space abs(1)_"dB" = 0 "dB"
$
Ora si calcolano le pulsazioni di taglio del polo e dello zero
$
  omega_z = 10/(C R_2) = 617 thin k"rad/"s space space omega_p = 1/(C R_2) = 61,7 thin k"rad/"s
$
#cfigure("images/Esame_08-02-2021_Disegno_2.png",50%)

Per il diagramma delle fasi teniamo a mente che
- uno #text(fill: green)[zero] negativo aumenta la fase di $90 degree$
- un #text(fill: red)[polo] negativo abbassa la fase di $90 degree$
- lo scostamento di fase avviene tra metà decade prima e metà decade dopo la pulsazione di taglio
- se il guadagno è positivo, il diagramma parte da $O degree$, se negativo parte da $- pi /2$


2) Calcolare il minimo valore di $R_L$ che garantisce staticamente la non saturazione dell'OPAMP sapendo che questo può erogare massimo $plus.minus 10 m A$ in uscita.

Siccome siamo in condizioni statiche $C$ è un circuito aperto, quindi il nostro circuito diventa

#cfigure("images/Esame_08-02-2021_Disegno_3.png",37%)

Abbiamo un OPAMP non invertente, quindi
$
  v_"OUT" = (1 + R_1/R_2) dot v_"IN" ==> v_"IN" = dfrac(R_1,R_1+R_2) dot v_"OUT"
$
Siccome non deve raggiungere la saturazione, dobbiamo considerare $v_"OUT" = plus.minus 10 V$; troviamo la formula per $i_"OUT"$, che sappiamo dalla consegna che deve essere $10 m A$

$
  i_"OUT" = i_1 + i_(R_L) =& dfrac( v_"OUT" , R_1 + R_2 ) + v_"OUT"/R_L 
  \
  ==>& R_L = dfrac( v_"OUT", i_"OUT" - dfrac( v_"OUT" , R_1 + R_2 ) ) 
$













=== Esame 17/06/2021 (Diagramma di Bode)
#cfigure("images/Esame_17-06-2021.jpg", 60%)
#cfigure("images/Esame_17-06-2021_Disegno_1.png", 60%)

#enum(
enum.item(1)[
  #text(fill: purple, size: 12pt)[Calcolo funzione di trasferimento]\
  OPAMP ideale ($L_+ = L_- = 0$) e in alto guadagno ($v_0=0$), quindi il circuito è lineare, e quindi si può applicare la sovrapposizione degli effetti.
  #circle(radius: 7pt)[#set align(center + horizon) 
                        1] $v_a != 0, v_B = 0 ==>$ Filtro attivo passa basso\
  #cfigure("images/Esame_17-06-2021_Disegno_2.png", 60%)
  $
    v_"OUT"_1 &= - dfrac(R,R) dot dfrac(1,1+ j omega C R) dot v_"IN" \
    &= - dfrac(1,1+ j omega C R) dot v_"IN"
  $

  #circle(radius: 7pt)[#set align(center + horizon) 
                        2] $v_a = 0, v_B != 0 ==>$ OPAMP non invertente\
  #cfigure("images/Esame_17-06-2021_Disegno_3.png",60%)
  $
  dfrac(1,Z_2) &= dfrac(1,R) + j omega C \
  &= dfrac(1+ j omega R C,R) ==> Z_2 = dfrac(R,1+ j omega R C)
  $

  Per calcolare la tensione $v_"BB"$ si usa la formula del partitore di tensione
  $
    v_"BB" = v_B dot dfrac(3R,3R+R)
  $

  $
    v_"OUT"_2 &= (1+ dfrac(Z_2,R) ) dot v_"BB" 
    \
    &= ( 1 + dfrac(Z_2,R) ) dot underbrace( v_B,v_"IN" ) dot underbrace( dfrac(3R, 3R + R), 3/4 ) 
    \
    &= dfrac(3,4) thin v_"IN" thin ( 1+ dfrac(1,1+j omega R C) )
  $

$
  v_"OUT" = v_"OUT"_1 + v_"OUT"_2 &=  - dfrac(1,1+ j omega C R) dot v_"IN" + dfrac(3,4) thin v_"IN" thin ( 1+ dfrac(1,1+j omega R C) )
  \
  &= v_"IN" ( - dfrac(1,1+ j omega C R) + dfrac(3,4) + dfrac(3,4 thin (1+j omega R C)) )
  \
  &= v_"IN" ( dfrac(-4 + 3 (1+j omega R C) + 3, 4 thin (1 + j omega R C)) )
  \
  &= v_"IN" thin dfrac(2 + 3 j omega R C, 4 thin (1 + j omega R C))
  \
  &= v_"IN" dfrac(1 + 3/2 j omega R C, 2 thin (1 + j omega R C))
  $
  quindi
  $
    H(j omega) = 1/2 thin dfrac(1 + 3/2 j omega R C, 1 + j omega R C)
  $

  Ora dobbiamo disegnare il diagramma di Bode della funzione di trasferimento appena ricavata.\
  Iniziamo con il diagramma del modulo:\
  - la funzione ha uno #text(fill: green)[zero], rappresentato da $1 + 3/2 j omega R C$
  - e un #text(fill: red)[polo], rappresentato da $1 + j omega R C$

  Possiamo utilizzare due metodi per tracciare il diagramma del modulo:
  - tracciare il diagramma del modulo dello #text(fill: green)[zero] e del #text(fill: red)[polo] e "sommarli"
  - calcolare il valore di $H(j omega)$ per $omega -> 0$ e $omega -> + infinity$ e trasformarlo in decibel
  Di seguito li illustrerò entrambi.

  $
    omega -> 0 ==> H(j omega) = 1/2 space space omega -> + infinity ==> H(j omega) = 3/4
    \
    abs(1/2)_"dB" = 20 log (1/2) = -6 space space abs(3/4)_"dB" = 20 log (3/4) = -2,5
  $

  Calcoliamo le pulsazioni di taglio di #text(fill: red)[polo] e #text(fill: green)[zero]
  $
    omega_p &= dfrac(1,R C) &space space omega_z &= dfrac(1, 3/2 R C)
    \
    &= 3030 "rad/"s & &=4545 "rad/"s
    \
    &= 3 thin k"rad/"s & &=4,5 thin k"rad/"s
  $

  Adesso possiamo disegnare il diagramma del modulo tenendo conto che prima di $omega_z$ vale $-6 "dB"$ e dopo $omega_p$ vale $-2,5 "dB"$, e che all'interno delle due pulsazioni di taglio aumenta con una pendenza di $20 "dB/dec"$

  #cfigure("images/Esame_17-06-2021_Disegno_4.png",35%)

  Per quanto riguarda il diagramma della fase anche qui si può tracciare il diagrammi di #text(fill: red)[polo] e #text(fill: green)[zero] e sommarli ma, siccome le pulsazioni di taglio si trovano nella stessa fase, il diagramma risultante dalla somma sarebbe impreciso, quindi dobbiamo utilizzare un altro metodo.

  Calcoliamo l'argomento della nostra funzione di trasferimento
  $
    arg( 1/2 thin dfrac( 1 + 3/2 j omega R C, 1 + j omega R C ) ) &= arg( 1 + 3/2 j omega R C ) - arg( 2 + 2 j omega R C )
  $

  $
    arg( 1 + 3/2 j omega R C ) = arctan( 3/2 w R C) &=
    cases(
      0 degree &wide omega << omega_z,
      45 degree &wide omega = omega_z,
      56 degree &wide omega = omega_p,
      90 degree &wide omega >> omega_z
    )
    \
    arg( 1 + j omega R C ) = arctan( w R C) &=
    cases(
      0 degree &wide omega << omega_p,
      -45 degree &wide omega = omega_p,
      -33 degree &wide omega = omega_z,
      -90 degree &wide omega >> omega_p
    )
  $

  quindi sommiamo tutte le quantità (tra l'altro si può immaginare senza difficoltà anche il valore che assume a metà tra le due pulsazioni di taglio)

  $
    arg( 1/2 thin dfrac( 1 + 3/2 j omega R C, 1 + j omega R C ) ) &=
    cases(
      0 degree &wide omega << omega_z wide omega << omega_p,
      12 degree &wide omega = omega_z,
      11.5 degree &wide omega = dfrac(omega_z + omega_p,2),
      11 degree &wide omega = omega_p,
      0 degree &wide omega >> omega_z wide omega << omega_p
    )
  $
  Quindi il diagramma delle fasi inizia metà decade prima di $omega_z$ a salire (da $0 degree$) fino a raggiungere il valore di circa $dfrac(pi,16)$ per tutto l'intervallo $[omega_z,omega_p]$, per poi scendere per metà decade dopo $omega_p$ e stabilizzarsi al valore $0 degree$.
],
enum.item(2)[
  #text(fill: purple, size: 12pt)[Calcolo $I_(O_"MAX")$]\
  Siccome siamo in condizioni statiche il circuito con $C$ è aperto, e nella relazione tra $v_"IN"$ e $v_"OUT"$ $omega=0$
  $
    v_"OUT" = 1/2 thin v_"IN"
  $

  $
    v_"IN" = -5 V ==> v_"OUT" = -2,5 V space space v_"IN" = 5 V ==> v_"OUT" = 2,5 V
  $

  $
    I_O = i_1 + I_L &= dfrac( v_"OUT"- v_"IN" , 2R ) + dfrac( v_"OUT", R_L )\
    &= 0,01654 thin A\
    &= 16,54 thin m A
  $
]
)



/*
=== Esercizio diagramma di Bode con OPAMP non ideale
Questo esercizio specifica di tracciare anche il diagramma di Bode considerando l'OPAMP non ideale, quindi ci da il #text(fill: purple)[prodotto guadagno-banda finito dell'OPAMP]
$
  A_v dot B = 1 M"Hz"
$
che bisogna trasformare in rad/s (basta moltiplicare per $2 pi$)
$
  A_v dot B = 6,28 "rad/"s #colred($:= omega_T$)
$
dove $omega_T$ è la #text(fill: red)[posizione della pulsazione al guadagno di taglio]. 

Prima si traccia il grafico del circuito ideale, dopo si modifica tale grafico in base alla pulsazione del guadagno di taglio. In particolare, per il diagramma del modulo, bisogna considerare una retta con pendenza $-20 "dB/dec"$ che interseca l'asse $log omega$; dopo il punto di intersezione tra il diagramma ideale e la retta, il diagramma reale segue l'andamento della retta; per il diagramma della fase, metà decade prima di $omega_T$ vi è uno sfasamento di $90 degree$.

#cfigure("images/Esercizio_Bode_non_ideale.png",60%)


Abbiamo la seguente funzione di trasferimento 
$
  H(j omega) = - 30 thin dfrac( 1+j omega C thin 35/3 thin R , 1+ j omega C thin 50 thin R )
$
$
  omega -> 0 ==> H(j omega) = -30 space space omega -> + infinity ==> H(j omega) = -5
  \
  abs(-30)_"dB" = 20 log (1/2) = 29 space space abs(-5)_"dB" = 20 log (3/4) = 14
$

*/



=== Esercizio Diagramma di Bode non ideale

#cfigure("images/Esercizio_2_Bode_non_ideale.png",70%)

*Amplificatore invertente*
$
  v_O_1 = - dfrac(R_2,R_1) dot v_"IN"
$ 

*Filtro attivo passa basso*
$
  v_O_2 = - dfrac(R_1,R_2) dot dfrac(1,1+j omega C R_4) dot v_"IN"
$

Applico la sovrapposizione degli effetti

#circle(radius: 7pt)[#set align(center + horizon) 
                        1] $v_O_1 != 0, v_O_2 = 0 $\
$
  v_"OUT"_1  = - dfrac(R_6,R_5) dot v_O_1 &= dfrac(R_6,R_5) dot dfrac(R_2,R_1) dot v_"IN"
  \
  &= 200 dot v_"IN" 
$

#circle(radius: 7pt)[#set align(center + horizon) 
                        2] $v_O_1 = 0, v_O_2 != 0 $\
Per calcolare $v_22$ utilizziamo la formula del partitore di tensione
$
  v_22 = v_O_2 dot dfrac(R_8,R_7+R_8)
$

$
  v_"OUT"_2 = (1 + R_6/R_5) dot v_22 &= (1 + R_6/R_5) (dfrac(R_8,R_7+R_8)) dot v_O_2
  \
  &= (1 + (20 cancel(R))/cancel(R)) (dfrac(R,6 R + R)) dot v_O_2
  \
  &=(1+20) dot 1/7 dot v_O_2
  \
  &= 3 v_O_2
$

$
  v_"OUT"_2 = -180 dot dfrac(1,1+j omega C R_4) dot v_"IN"
$

$
  v_"OUT" = 200 dot  v_"IN" - 180 dfrac(1,1+j omega 60 C R) dot v_"IN"
  = 20 dfrac(1 + j omega 600 C R, 1 + j omega 60 C R) dot v_"IN"
$
$
  H(j omega) = 20 dfrac(1 + j omega 600 C R, 1 + j omega 60 C R)
$

Questo esercizio specifica di tracciare anche il diagramma di Bode considerando l'OPAMP non ideale, quindi ci da il #text(fill: purple)[prodotto guadagno-banda finito dell'OPAMP]
$
  A_v dot B = 1 M"Hz"
$
che bisogna trasformare in rad/s (basta moltiplicare per $2 pi$)
$
  A_v dot B = 6,28 "rad/"s 
$
da cui si ricava $omega_H$, cioè la posizione della pulsazione al guadagno di taglio
$
  omega_H = dfrac(A_v dot B, abs(mu dot dfrac("coeff zero", "coeff polo") ))
$

Prima si traccia il grafico del circuito ideale, dopo si modifica tale grafico in base alla pulsazione del guadagno di taglio. In particolare, per il diagramma del modulo, bisogna considerare una retta con pendenza $-20 "dB/dec"$ che interseca l'asse $log omega$; dopo il punto di intersezione tra il diagramma ideale e la retta, il diagramma reale segue l'andamento della retta; per il diagramma della fase, metà decade prima di $omega_T$ vi è uno sfasamento di $90 degree$.

$
  w -> 0 ==> H(j omega) tilde 20 = 26 "dB" space space w -> + infinity ==> H(j omega) tilde 200 = 46 "dB"
$

$
  omega_z = dfrac(1,600 C R)= 297,6 "rad/"s space space omega_p = dfrac(1,60 C R)= 2976 "rad/"s 
  \
  omega_H = dfrac(A_v dot B, abs(20 dot 600/60))= 31,4 thin k"rad/"s
$

#cfigure("images/Esercizio_2_Bode_non_ideale_2.png",80%)














=== Esame 09/11/2021 (frequenza di taglio)
Dal seguente circuito si calcolino i valori di $R_1$ e $R_2$ in modo che la frequenza di taglio sia $20 "Hz"$ e il guadagno in centro banda risulti pari a 10. Si supponga l'OPAMP ideale e in alto guadagno.

#cfigure("images/Esame_9-11-2021_Disegno_1.png", 55%)

Dati:
$
  R_3 &= 9 thin k ohm &space f_T &= 20 "Hz" \
  C &= 1 thin mu F & A_(v_"CB")&= 10 \
  L_+ = - L_- &= 10 V
$

#text(fill: green, size: 12pt)[OPAMP non invertente]
$
  v_"OUT" &= (1+ R_3/R_2) dot v_x \
  &= ( dfrac( R_2 + R_3, R_2 ) ) dot v_x
$

#text(fill: purple, size: 12pt)[Partitore di tensione]
$
  v_x &= v_"IN" dot dfrac( R_1, Z_C + R_1 ) \
  &= v_"IN" dot dfrac( R_1, dfrac(1,j omega C) + R_1 ) \
  &= v_"IN" dot dfrac( R_1, dfrac(1 + j omega C R_1,j omega C)) \
  &= v_"IN" dot dfrac( j omega C R_1, 1 + j omega C R_1 )
$

Quindi la funzione di trasferimento è:
$
  H(j omega) = underbrace( dfrac(R_2 + R_3, R_2) , mu ) dot dfrac( j omega C R_1, 1 + j omega C R_1 )
$
$
  abs(A_(v_"CB"))_"dB" = 20 log_10 (10) = 20 "dB"
$

#cfigure("images/Esame_9-11-2021_Disegno_2.png", 45%)

$
  omega_T = 1/(R_1 C) space omega_T = 2 pi f_T wide ==> wide  R_1 = 1/(omega_T C) &= dfrac(1,2 pi f_T C)
  \
  &= dfrac( 1 , 2 pi dot 20 "Hz" dot 1 times 10^(-6) F )
  \
  &= 7,96 thin k ohm
$

$
  abs(A_(v_"CB"))_"dB" = mu  = dfrac(R_2 + R_3 , R_2) = 10 ==> &R_2 + R_3 = 10 dot R_2
  \
  &R_2 - 10 dot R_2 = - R_3
  \
  &-9 dot R_2 = - R_3
  \
  &9 R_2 = R_3 
  \
  &R_2 = R_3/9 = 1 thin k ohm
$



=== Esame 30/06/2021 (grafico corrente)
Si deve tracciare nel piano la relazione $I_"IN""-"I_"OUT"$ per $I_"IN" in [-10 m A thin .. thin 10 m A ]$.

La relazione trovata è
$
  I_"OUT" = -0,1 dot I_"IN" + 0,4 m A
$

Calcoliamo i valori interessanti di $I_"OUT"$ in relazione a $I_"IN"$ e in relazione ai valori di saturazione $L_+ = L_- = 10 V$, tenendo conto che $v_"OUT" = - 1 k ohm dot I_"IN" + 4 V$
$
  I_"IN" &= -10 m A &space space I_"OUT" &= 1,4 m A
  \
  I_"IN" &= 10 m A & I_"OUT" &= -0,6 m A
  \
  I_"IN" &= 0 m A &space space I_"OUT" &= 0,4 m A
  \
  L_+: space - 1 k ohm dot I_"IN" + 4 V &=10 V & I_"IN" &= -6
$

#cfigure("images/Screenshot_20240910-101122.png",55%)







== D 
=== Formule notevoli
$
  C_(min) = "Cox" dot L_(min)^2 dot ("SP" + "SN") \
   ln(2) dot R_("eq") dot C_"INV" <= t \
  "Resistenza equivalente pull-up" space R_("eq P") &= t_("LH")/(ln(2) dot C_(min)) wide  R_("eq P") = dfrac(t_"LH",ln(2) dot C_"INV")  \
  "Resistenza equivalente pull-down" space R_("eq N") &= t_("HL")/(ln(2) dot C_(min)) wide R_("eq N") = dfrac(t_"HL",ln(2) dot C_"INV") \
  R_(P n) &= (R_("eq P") - dfrac(R_("RIF" P),S_P) dot N )/K \
  "Per percorsi critici" space R_P &= R_("eq P")/K\
  S_(P) &= R_("RIF" P)/(R P)
$
*Note:* 
- $ln(2) = 0,69$
- la $S_P$ che compare nella formula di $R_(P n)$ è sempre quella del percorso critico 
- $t_("LH")$ è il tempo di salita e $t_("HL")$ è il tempo di discesa. In generale negli esercizi se chiede di "dimensionare affinché il tempo di salita al nodo $X$ sia inferiore o uguale a $Y p s$" vuol dire che prenderemo $t_("LH") = Y$.
- $p s$ sono pico secondi
- quando non vengono dati SP e SN utilizzare le altre formule della resistenza equivalente



=== Regole generali per la rete di pull-up e pull-down
Abbiamo questa rete
#cfigure("images/240909_18h24m35s_screenshot.png",30%)
Conoscendo la funzione $O$, e sapendo che $X = overline(O)$:
- la rete di pull-up sarà uguale a la $X$, senza il negato e con somma e prodotto invertiti
- la rete di pull-down sarà uguale a la $X$ senza il negato
- gli elementi in serie sono il prodotto booleano degli elementi
- gli elementi in parallelo sono la somma booleana deli elementi

Se la $O = overline(A overline(C) + C overline(A) + B overline(D))$, allora
$
  X = overline(O) &= overline( overline(A overline(C)) dot overline(C overline(A)) dot overline(B overline(D)) )
  \
  &=overline( (overline(A) + C) dot (overline(C) + A) dot (overline(B) + D) )
$
Rete di pull-up:
$
  overline(A) C + overline(C) A + overline(B) D
$ 

Rete di pull-down:
$
  (overline(A) + C) dot (overline(C) + A) dot (overline(B) + D)
$

#cfigure("images/240909_18h34m03s_screenshot.png",40%)



=== Esame 14/06/2023
+ Della rete in figura si calcoli l'espressione booleana al nodo O.
+ Dimensionare i transistori pMOS affinché il tempo di salita al nodo F sia inferiore o uguale a 90ps. Ottimizzare il progetto. Si tenga conto che i transistori dell'inverter di uscita hanno le seguenti geometrie : Sp = 200, Sn = 100.
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
  - Gli elementi in serie sono il prodotto booleano degli elementi
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
  *N.B.* Bisogna specificare $overline(B)$ e non $X$ perché si deve considerare solo il percorso di $overline(A) C$, e se $B$ fosse accesso il percorso sarebbe diverso.

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

  *Nota:* le $S_P$ trovate denotano la dimensione massima dei transistori che interessano il percorso; in particolare si assegna prima la dimensione ai transistori presenti nel percorso critico, poi agli altri, in modo che il valore trovato per un transistori del percorso critico sia dominante rispetto al valore trovato per lo stesso transistore per un percorso non critico.

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

    Quindi per tutti i percorsi critici individuati $K=3,5$

    $
      R_N = dfrac(R_("eq" N),K) &= dfrac(736 thin ohm,3.5) &space S_N = dfrac(R_("RIF" N), R_N) &= dfrac(5000 thin ohm,210 thin ohm) \
      &= 210 thin ohm &space &=24
    $

    Tutti i transistori della rete pull-down avranno quindi dimensione 24

    #cfigure("images/2024-08-07-18-12-47.png", 50%)




=== Esame 28/01/2021

2) 
$
  t_"LH" = t_"HL" = 100 p s  
$
$
  R_("EQ"_N) = dfrac(t_"HL", ln (2) dot C_"INV") &= dfrac(100 dot 10^(-12) s , 0.69 dot 90 dot 10^(-15)  )
  \
  &= 1610 ohm
$

*Primo caso peggiore - percorso critico $bold( overline(A) B C overline(D) "CLK" )$*

In questo caso $K = 4$ (numero di nMOS)

$
  R_N = R_"EQ"_N/K &= dfrac( 1610 ohm , 4 ) &space space S_N &= dfrac( 5000 ohm , 402.5 ohm )
  \
  &= 402.5 ohm & &=12.43 tilde.equiv 13
$

*Secondo caso peggiore $bold( A overline(B) overline(C) D "CLK" )$*
$ K = 2, N = 1$

$
  R_N_2 = dfrac( R_"EQ"_N - R_"RIF"_N/S_N dot N,K ) 
  &= dfrac( 1610 ohm - dfrac( 5000 ohm,13 ), 2 ) &space space S_N &= dfrac( 5000 ohm , 612.7 ohm )
  \
  &= 612.7 ohm & &= 8.16 tilde.equiv 9
$

*Terzo caso peggiore $bold( A B overline(C) D "CLK" )$*


Per passare per D bisogna "aprire" anche A, ma in questo modo la corrente, tra il percorso BD e il percorso A in parallelo, sceglie sempre di passare da A, perché fa un solo salto, e non passa mai da D. Per questo diamo a D la dimensione minima
$
  D = 1
$

























