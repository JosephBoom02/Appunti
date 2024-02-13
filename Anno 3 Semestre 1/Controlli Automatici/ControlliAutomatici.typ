
#import "@preview/physica:0.9.0": *
#import "@preview/i-figured:0.2.3"
#import "@preview/cetz:0.1.2" 
#import "@preview/xarrow:0.2.0": xarrow


#import cetz.plot 

#let title = "Controlli Automatici T"
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

#let space = h(5em)

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

//Justify text



= Introduzione 
L'idea dei #text(weight: "bold")[controlli automatici] è sostituire l'intelligenza umana con un sistema automatico (come l'intelligenza artificiale) basata su leggi matematiche e/o algoritmi.



== Notazione ed elementi costitutivi

#figure(image("Images/Schema_sistema.png" ,width: 50%))

Il #text(weight: "bold")[sistema] è un oggetto per il quale si vuole ottenere un comportamento desiderato.


Esempi di sistema sono: impianto (industriale), macchinario (braccio robotico, macchina a controllo numerico, etc ...), veicolo (auto, velivolo, drone, etc ...), fenomeno fisico (condizioni atmosferiche), sistema biologico, sistema sociale.



L'obiettivo è che l'andamento nel tempo di alcune variabili segua un segnale di riferimento.

Altri elementi sono:

- Controllore: unità che determina l'andamento della variabile di controllo (ingresso);
- Sistema di controllo: sistema (processo) + controllore;
- Sistemi di controllo naturali: meccanismi presenti in natura, come  quelli presenti nel corpo umano (temperatura corporea costante, ritmo cardiaco, etc dots);
- Sistemi di controllo manuali: è presente l'azione dell'uomo;
- Sistemi di controllo automatico: uomo sostituito da un dispositivo.




== Controllo in anello aperto e anello chiuso
Controllo in anello aperto *“feedforward”*: il controllore utilizza solo il segnale di riferimento

#figure(image("Images/Anello_aperto.png", width: 75%))

#text(weight: "bold")[Controllo in anello chiuso] (“*feedback*” o retroazione): il controllore utilizza il segnale di riferimento e la variabile controllata ad ogni istante di tempo

#figure(image("Images/Anello_chiuso.png", width: 80%))

Il controllo in retroazione è un paradigma centrale nei controlli automatici.


== Progetto di un sistema di controllo
I passi per progettare un sistema di controllo sono:

- definizione delle specifiche: assegnazione comportamento  desiderato, qualità del controllo, costo,...
- modellazione del sistema (controllo e test): complessità del modello (compromesso), definizione ingressi/uscite, codifica del modello, validazione in simulazione
- analisi del sistema: studio proprietà “strutturali”, fattibilità specifiche
- sintesi legge di controllo: è basata su modello, analisi sistema controllato, stima carico computazionale
- simulazione sistema controllato: test su modello di controllo, test realistici (modello complesso, ritardi, quantizzazione, disturbi, ...)
- scelta elementi tecnologici: sensori/attuatori, elettronica di acquisizione/attuazione, dispositivo di elaborazione
- sperimentazione: hardware in the loop, prototipazione rapida, realizzazione prototipo definitivo






== Esempio di sistema di controllo: circuito elettrico <Circuito_elettrico>

#figure(image("Images/Es_circuito_elettrico.png", width: 35%))

La legge che usiamo per definire il circuito (il nostro sistema) è la #text(style: "italic")[legge delle tensioni]

$ v_R (t) = v_G (t) - v_C(t) $

le leggi del condensatore e del resistore sono 

$ C dot dot(v)_C (t) = i(t) #h(2cm) v_R (t) = R dot i(t) $    

Scrivendo la formula in termini di $v_C (t)$ (stato interno) e $v_G (t)$ (ingresso di controllo)

$ dot(v)_C (t) = 1 / "RC" (v_G (t) - v_C (t)) $   



= Sistemi in forma di stato
== Sistemi continui
I #text(style: "italic")[sistemi continui] sono sistemi in cui il tempo è una variabile reale: $t in RR$

$ 
dot(x)(t) &= f ( x(t), u(t), t ) #h(3em) && "equazione di stato" \
dot(y)(t) &= h(x(t), u(t), t)  && "equazione (trasformazione) di uscita"
$ <sistemi_continui>

Definiamo inoltre $t_0$ come tempo iniziale e $x(t_0)=x_0$ come stato iniziale.

#text(weight: "bold")[N.B.] $dot(x)(t) := display(frac(d, d t)) x(t)$.

Notazione:
- $x(t)  in RR^n$ stato del sistema all'istante $t$
- $u(t)  in RR^m$ ingresso del sistema all'istante $t$
- $y(t)  in RR^p$ uscita del sistema all'istante $t$

$
    x(t)=
    mat(delim: "[",
        x_1(t);
        dots.v;
        x_n(t))
    #h(3.5em)
    u(t) =
    mat(delim: "[",
        u_1(t);
        dots.v;
        u_m(t))
    #h(3.5em)
    y(t) = 
    mat(delim: "[",
        y_1(t);
        dots.v;
        y_p(t))
$
Da notare che $x(t)$ è un vettore mentre $x_1,...,x_n$ sono scalari; $x(t)$ è una variabile interna che descrive il comportamento del sistema.


===  Equazione di stato
L #text(style: "italic")[equazione di stato] è un'equazione differenziale ordinaria (ODE) vettoriale del primo ordine (cioè l'ordine massimo delle derivate è 1) 
$
dot(x)_1(t) &= f_1 (mat(delim: "[", x_1 (t); dots.v; x_n (t)), mat(delim: "[", u_1 (t); dots.v; u_m (t)), t) \
&dots.v \
dot(x)_n (t) &= f_n (mat(delim: "[", x_1 (t); dots.v; x_n (t)), mat(delim: "[", u_1 (t); dots.v; u_m (t)), t)
$
$RR^n$ è detto #underline[spazio di stato], con $n$ ordine del sistema. La funzione di stato è $f: RR^n times RR^m times RR -> RR^n$.
$
    mat(delim: "[",
        dot(x)_1(t);
        dots.v;
        dot(x)_n(t);
    ) 
    =
    mat(delim: "[",
        f_1 (x(t),u(t),t);
        dots.v;
        f_n (x(t),u(t),t);
    )
    := f (x(t),u(t),t)
$
Avere solo derivate prime non è limitato, perché ad esempio posso inserire una prima variabile come derivata prima e una seconda variabile come derivata prima della prima variabile.


===  Equazione di uscita
L'equazione di uscita è un'equazione algebrica
$
y_1(t) &= h_1 (x(t), u(t), t) \
&dots.v \
y_p (t) &= h_p (x(t), u(t), t)
$
$h : RR^n times RR^m , RR -> RR^p$ funzione di uscita
$
    mat(delim: "[",
        y_1(t);
        dots.v;
        y_p (t)
    ) 
    =
    mat(delim: "[",
        h_1 (x(t),u(t),t);
        dots.v;
        h_p (x(t),u(t),t)
    )
    := h(x(t),u(t),t)
$



Se la soluzione $x(t)$ a partire da un istante iniziale $t_0$ è univocamente determinata da $x(t_0)$ e $u(tau)$ con $tau >= t_0$, allora il sistema è detto #text(weight: "bold")[causale], cioè lo stato dipende solo da ciò che accede in passato.

Sotto opportune *ipotesi* di regolarità della funzione $f$ si dimostra esistenza e unicità della soluzione dell'equazione (differenziale) di stato (Teorema di Cauchy-Lipschitz).



== Sistemi discreti
Nei _sistemi discreti_ il tempo $t$ è una variabile interna, $t in ZZ$.
$
    x(t+1) &= f  (x(t), u(t), t ) #h(3.5em) && "(equazione di stato)"  \
    y(t) &= h (x(t), u(t), t ) && "(equazione (trasformazione) di uscita)"
$ <sistemi_discreti>
L'equazione di stato è un'equazione alle differenze finite (FDE).

Notazione:
- $x(t) in RR^n$ stato  del sistema all'istante $t$
- $u(t) in RR^m$ ingresso del sistema all'istante $t$
- $y(t) in RR^p$ uscita del sistema all'istante $t$



$x(t),u(t)" e " y(t)$ sono uguali ai sistemi continui.

Per modellare sistemi discreti nel codice basta un ciclo ```matlab for ```.



== Esempio circuito elettrico
Riprendiamo l'esempio del @Circuito_elettrico[circuito elettrico]; la formula trovata è 
$
    underbrace(dot(v)_C(t), dot(x)(t)) = frac(1,R C) lr((underbrace(v_G (t),u(t)) - underbrace(v_C (t),x(t))), size: #35%)
$
In questo caso lo stato del sistema $x(t)$ è caratterizzato dalla variabile $v_C (t)$, l'ingresso dalla variabile $v_G (t)$. Supponiamo quindi di misurare (con un sensore) la tensione ai capi della resistenza, allora l'uscita del nostro sistema sarà $v_R (t)$
$
    dot(x)(t) &= frac(1, R C) (u(t)-x(t)) #h(4em)
    f(x,u) &&= frac(1, R C)(u-x)
$
da notare che in questo caso $f$ non è funzione del tempo.
$
    v_R (t) = v_G (t) - v_C (t) ==> y(t) = u(t) - x(t)
$


===  Esempio con parametri che variano nel tempo
Supponiamo che la resistenza sia una funzione del tempo
$
    R(t) = overline(R)  (1- frac(1,2) e^(-t) )
$
allora 
$
    dot(x)(t) &= frac(1,R(t)C)  (u(t)-x(t) ) #h(3.5em)
    f(x,u,t) &= frac(1,R(t)C)(u-x)
$
in questo caso $f$ è funzione del tempo.



== Esempio carrello
#figure(image("Images/Es_carrello.png", width: 40%))
La legge che usiamo è la legge di Newton, prendendo $z$ come posizione del centro di massa
$
    M  dot.double(z)   = -F_e + F_m
$
con $M$ massa e $F_e$ data da
$
    F_e (z(t), t) = k(t)z(t)
$
quindi la nostra equazione diventa
$
    M dot.double(z)(t) = -k(t)z(t) + F_m (t)
$
Siccome nella nostra formula compare una derivata seconda di una variabile ci conviene definire lo stato del sistema con la variabile stessa e la derivata prima della variabile.

Definiamo quindi $x_1 := z$ e $x_2:=dot(z)$, con stato $x := [x_1x_2]^T$, e $u := F_m$ (ingresso).

Quindi possiamo scrivere, tenendo conto che $dot(x)_2(t) = dot.double(z)$
$
     dot(x)_1(t) &= x_2(t) \
     dot(x)_2(t) &= - frac(k,M) x_1(t) +  frac(u(t),M)
$
$
    f(x,u) = 
    mat(delim: "[",
        f_1(x,u);
        f_2(x,u);
    )
    :=
    mat(delim: "[",
        x_2;
        - display(frac(k,M))x_1+ display(frac(u,M));
    )
$

Supponiamo di misurare $z(t)$ (sensore posizione), allora $y := z$
$
    dot(x)_1(t) &= x_2(t) \
    dot(x)_2(t) &= - frac(k,M) x_1(t) +  frac(u(t),M)  \
    y(t) &= x_1(t)
$

Sia $k(t) = k$ e, ricordando la formula dell'energia cinetica $E_(k)=display(frac(1,2)) m v^(2)$ e la formula dell'energia elastica $U= display(frac(1,2)) k Delta x^2$, consideriamo come uscita l'energia totale $E_T (t) =  display(frac(1,2)) (k z^2 (t) + M  dot(z)^2 (t))$
$
     dot(x)_1(t) &= x_2(t) \  
     dot(x)_2(t) &= - frac(k,M) x_1(t) +  frac(u(t),M) \
    y(t) &=  frac(1,2)  (k(t) x_1^2 (t) + M  x_2^2 (t) )
$
quindi $h(x):= dfrac(1,2) (k x_1^2 + M x_2^2)$.

*N.B.* Il risultato (l'uscita) vale, di solito, solo per il mio modello, in base a come l'ho impostato; nella realtà potrebbe essere diverso.



== Esempio auto in rettilineo
#cfigure("Images/Es_Rettilineo.png", 60%)

Scriviamo la legge di Newton
$
    M  dot.double(z) = F_"drag" + F_m
$
con $M$ massa e $F_"drag"$ data da
$
    F_("drag") = -b  dot(z)
$
Definiamo $x_1 := z$ e $x_2 :=  dot(z)$ $("stato" x := [x_1 x_2 ]^T )$ e $u := F_m$ (ingresso). Supponiamo di misurare $z(t)$ (sensore posizione), allora $y := z$
$
    dot(x)_1(t) &= x_2(t) \
    dot(x)_2(t) &= -  frac(b,M) x_2(t) +  frac(1,M)u(t) \ 
    y(t) &= x_1(t)
$

Proviamo a progettare un sistema per il _cruise control_.  
L'equazione della dinamica è
$
    M  dot.double(z)(t) = -b dot(z)(t) + F_m (t)
$
Siccome siamo interessati a controllare la velocità e non la posizione, allora consideriamo come stato solo la velocità: $x :=  dot(z)$, $u := F_m$. Supponiamo di misurare $ dot(z)(t)$ (sensore velocità), allora $y := x$
$
    dot(x)(t) &= -frac(b,M)x(t) +  frac(1,M)u(t)  \
    y(t) &= x(t)
$



== Esempio pendolo
#cfigure("Images/Es_pendolo.png", 25%)

Scriviamo l'equazione dei momenti
$
    M  ell^2  dot.double(theta)= C_"grav" + C_"drag" + C_m
$
con $M$ massa e $C_"grav"$ e $C_"drag"$ date da
$
    C_"grav" &=M g  ell  sin( theta) & #h(5em)
    C_"drag" &= -b  dot  theta
$
con $b$ coefficiente d'attrito.

Scriviamo l'equazione della dinamica, partendo dalla formula iniziale dei momenti
$
     dot.double(theta)(t) = - frac(g, ell)  sin  ( theta(t) ) -  frac(b,M  ell^2)  dot(theta)(t) +  frac(1,M  ell^2) C_m (t)
$
Definiamo quindi $x_1 :=  theta$ e $x_2 :=  dot(theta)$ (stato $x:= [x_1x_2]^T$) e $u := C_m$ (ingresso).  
Supponiamo di misurare $ theta$ (sensore angolo) , allora $y :=  theta$

$
    dot(x)_1 (t) &= x_2(t) \
    dot(x)_2(t) &= - frac(g, ell)  sin  (x_1(t) ) -  frac(b,M  ell^2) x_2(t) +  frac(1,M  ell^2) u(t)  \
    y(t) &= x_1(t)
$
Se misuriamo invece la posizione verticale, allora $y := -  ell  cos(theta)$ 
$
    dot(x)_1 (t) &= x_2(t)  \
    dot(x)_2(t) &= - frac(g, ell)  sin  (x_1(t) ) -  frac(b,M  ell^2) x_2(t) +  frac(1,M  ell^2) u(t)  \
    y(t) &= -  ell  cos( theta)
$



== Traiettoria di un sistema
<traiettoria_di_un_sistema>
Dato un istante iniziale $t_0$ e uno stato iniziale $x_(t_0)$, la funzione del tempo $(x(t), u(t)),   t>t_0$, che soddisfa l'equazione di stato $ dot(x)(t) = f (x(t), u(t), t)$ si dice traiettoria (movimento) del sistema. In particolare, $x(t)$ si dice traiettoria dello stato. Consistentemente, $y(t)$ si dice traiettoria dell'uscita.

*N.B.* per sistemi senza ingresso (quindi non forzati) la traiettoria dello stato $x(t),   t>t_0$ è determinata solo dallo stato iniziale $x_(t_0)$.


===  Esempio
Definiamo un sistema con stato $x$ e stato iniziale $x_0$
$
    x &:=
    mat(delim: "[",
        x_1;
        x_2;
    )
    #h(5em)
    x_0 &:=
    mat(delim: "[",
        5;  
        3;
    )
    #h(5em)
    t_0 &= 0
$
$
    dot(x)_1(t) &= x_2(t)  \
    dot(x)_2(t) &= u(t)
$
Assegno a $x_1$, $x_2$ e $u(t)$ le seguenti equazioni
$
    overline(x)_1(t) &= 5+3t+t^2  \
    overline(x)_2(t) &= 3+2t  \
    overline(u)(t) &= 2
$
Se le equazioni di $ overline(x)_1$ e $ overline(x)_2$ soddisfano le condizioni iniziali e la funzione di stato ($dot(x)_1$ e $dot(x)_2$) allora quelle equazioni sono la traiettoria del sistema.  
Infatti

$
    & overline(x_0) = 
    mat(delim: "[",
        5+3t+t^2; 
        3+2t
    )_(t=0) 
    =
    mat(delim: "[",
        5;
        3
    )
    #h(5em)
    frac(d,d t) 
    mat(delim: "[",
        5+3t+t^2;
        3+2t
    )
    =
    mat(delim: "[",
        3+2t;
        2
    )
$



== Equilibrio di un sistema
Dato un #underline[sistema (non forzato)] $dot(x)(t) = f (x(t), t)$, uno stato $x_e$ si dice _equilibrio del sistema_ se $x(t) = x_e$ , $t >= t_0$ è una traiettoria del sistema.

Dato un #underline[sistema (forzato)] $dot(x)(t) = f (x(t), u(t), t)$, $(x_e , u_e )$ si dice _coppia di equilibrio_ del sistema se $(x(t), u(t)) = (x_e , u_e )$, $t  >= t_0$ , è una traiettoria del sistema.
 
Per un #underline[sistema (tempo invariante continuo)] $dot(x)(t) = f (x(t), u(t))$ data una coppia di equilibrio $(x_e,u_e)$ vale $f(x_e,u_e)=0$.  
Se il sistema è non forzato, dato un equilibrio $x_e$ vale $f(x_e)=0$.


===  Esempio pendolo

$
    dot(x)_1(t) &= x_2(t) &= f_1(x(t),u(t))  \
    dot(x)_2(t) &= -  frac(G, ell)  sin (x_1(t)) -  frac(b,M  ell ^2)x_2(t) +  frac(1,M ell^2)u(t) #h(4em) &=f_2(x(t),u(t))
$
Siccome sappiamo che, data una coppia di equilibrio $(x_e,u_e)$, vale $f(x_e,u_e)=0$, allora per trovare l'equilibrio del pendolo imponiamo 
$
    f(x_e,u_e)=0
$
cioè:
$
    cases(
        x_(2e)(t) = 0 \
        -  dfrac(G, ell)  sin (x_(1e)) -  dfrac(b x_(2e),M  ell ^2) +  dfrac(1,M ell^2)u_e =0
    )
$ 
sostituendo $x_(2e)(t)=0$ nell'ultima equazione
$
    -  dfrac(G, ell)  sin (x_(1e)) +  dfrac(1,M ell^2)u_e =0  ==> u_e = M G  ell  sin(x_(1e))
$
In conclusione, le coppie di equilibrio del sistema sono tutti gli $(x_(1e), x_(2e),u_e)$ che soddisfano

$
    cases(
        u_e = M G  ell  sin(x_(1e))  \
        x_(2e)=0
    )
$




== Classificazione dei sistemi in forma di stato
La classe generale è  $x  in RR^n , u  in RR^m , y in RR^p$
$
    dot(x)(t) &= f (x(t), u(t), t) &space & "equazione di stato"\ 
    y(t) &= h(x(t), u(t), t) & & "equazione di uscita"
$ <sistemi_forma_stato> 

    - I sistemi *monovariabili* (SISO, Single Input Single Output) sono una sottoclasse di sistemi *multivariabili* (MIMO, Multiple Input Multiple Output); sono tali se $m=p=1$, altrimenti sono dei sistemi MIMO;
    - I sistemi *strettamente propri* sono una sotto classe dei *sistemi propri*; sono tali se $y(t) = h(x(t),t)$, quindi se l'uscita dipende esclusivamente dall'ingresso, chiamati quindi sistemi causali (tutti i sistemi che abbiamo visto fin'ora sono sistemi propri).
    - I sistemi *non forzati* sono una sotto classe dei *sistemi forzati*; un esempio di sistema non forzato è il seguente
    $
        dot(x)(t) &= f(x(t),t) \
        y(t) &= h(x(t),t)
    $
    - I sistemi *tempo invarianti* sono una sotto classe di sistemi *tempo  varianti*; sono sistemi in cui le funzioni $f$ e $h$ #underline[non dipendono esplicitamente] dal tempo, cioè risulti
    $
        dot(x)(t) = f(x(t), u(t)) \
        y(t) = h(x(t), u(t))
    $  
    I tempo invarianti sono tali se, data una traiettoria $ (x(t), u(t)), t >= t_0$, con $x(t_0)=x_0$, per ogni $ Delta  in RR$ vale che $x(t_0+ Delta)=x_0$ allora $(x_( Delta) (t), u_( Delta) (t)) = (x(t- Delta), u(t- Delta))$ è una traiettoria.  
    Si può dimostrare che sistemi tempo invarianti sono del tipo
    $
        dot(x)(t) &= f (x(t), u(t)) &space x(0)=x_0 \
        y(t) &= h(x(t), u(t))
    $
    e senza senza perdita di generalità possiamo scegliere $t_0=0$.  
    Graficamente:
    #figure(image("Images/Sistemi_tempo_invarianti.png", width: 65%))
    - I *sistemi lineari* sono una sotto classe di *sistemi non lineari*.  
    I sistemi lineari sono tali se le funzioni di stato e di uscita sono lineari in $x$ e $u$:
    $
        dot(x)_1 (t) &= a_(11) (t)x_1 (t) + a_(12) (t)x_2 (t) + . . . + a_(1n) (t)x_n (t)+ b_(11) (t)u_1 (t) + b_(12) (t)u_2 (t) + . . . + b_(1m) (t)u_m (t)  \
        dot(x)_2 (t) &= a_(21) (t)x_1 (t) + a_(22) (t)x_2 (t) + . . . + a_(2n) (t)x_n (t)+ b_(21) (t)u_1 (t) + b_(22) (t)u_2 (t) + . . . + b_(2m) (t)u_m (t)  \
        &dots.v \ 
        dot(x)_n (t) &= a_(n 1) (t)x_1 (t) + a_(n 2) (t)x_2 (t) + . . . + a_(n n) (t)x_n (t)+ b_(n 1) (t)u_1 (t) + b_(n 2) (t)u_2 (t) + . . . + b_(n m) (t)u_m (t)
    $
    per $y(t)$ invece 
    $
        y_1 (t) &= c_(11) (t)x_1 (t) + c_(12) (t)x_2 (t) + . . . + c_(1n) (t)x_n (t)+ d_(11) (t)u_1 (t) + d_(12) (t)u_2 (t) + . . . + d_(1m) (t)u_m (t)  \
        y_2 (t) &= c_(21) (t)x_1 (t) + c_(22) (t)x_2 (t) + . . . + c_(2n) (t)x_n (t)+ d_(21) (t)u_1 (t) + d_(22) (t)u_2 (t) + . . . + d_(2m) (t)u_m (t)  \
        &dots.v \ 
        y_p (t) &= c_(p 1) (t)x_1 (t) + c_(p 2) (t)x_2 (t) + . . . + c_(p n) (t)x_n (t)+ d_(p 1,t)u_1 (t) + d_(p 2) (t)u_2 (t) + . . . + d_(p m) (t)u_m (t)
    $




== Proprietà dei sistemi lineari
===  Sistemi lineri in forma matriciale
Definiamo le matrici $A(t)  in RR^(n  times n) , B(t)  in RR^(n  times m) , C(t)  in RR^(p  times n) , D(t)  in RR^(p  times m)$
$
    A(t) &= mat(delim: "[",
        a_(11,t) , ... , a_(1n,t);
        dots.v;
        a_(n 1,t) , ... , a_(n n,t)
    )
    &#h(5em)
    B(t) &= mat(delim: "[",
        b_(11,t) , ... , b_(1m,t) ; 
        dots.v;
        b_(n 1,t) , ... , b_(n m,t)
    )
    \
    C(t) &= mat(delim: "[",
        c_(11,t) , ... , c_(1n,t)  ;
        dots.v;
        c_(p 1,t) , ... , c_(p n,t)
    )
    &
    D(t) &= mat(delim: "[",
        d_(11,t) , ... , d_(1m,t)  ;
        dots.v;
        d_(p n 1,t) , ... , d_(p m,t)
    )
$ <matrici_sistemi_lineari>
quindi scriviamo
$
    mat(delim: "[",
        dot(x)_1(t) ; 
        dots.v;
        dot(x)_ n(t)
    )
    = A(t)
    mat(delim: "[",
        x_1(t)  ;
        dots.v;
        x_n (t)
    )
    + B(t)
    mat(delim: "[",
        u_1 (t)  ;
        dots.v;
        u_m (t)
    )
    \
    mat(delim: "[",
        y_1 (t) ; 
        dots.v;
        y_n (t)
    )
    = C(t)
    mat(delim: "[",
        x_1(t)  ;
        dots.v;
        x_n (t)
    )
    + D(t)
    mat(delim: "[",
        u_1(t)  ;
        dots.v;
        u_m (t)
    )
$ <matrici_sistemi_lineari_2>
che equivale a 
$
    dot(x)(t) &= A(t) x(t) + B(t) u(t)  \
    y(t) &= C(t)x(t) + D(t) u(t)
$ <sistema_lineare>


== Sistemi lineari tempo-invarianti
I _sistemi lineari tempo invarianti_ sono sistemi lineari in cui le matrici $A,B,C,D$ sono matrici costanti.
$
    dot(x)(t) = A x(t) + B u(t)  \
    y(t) = C x(t) + D u(t)
$ <sistemi_lineari_tempo_invarianti>


===  Esempio carrello
#figure(image("Images/Es_carrello.png", width: 40%))

$
    dot(x)_1(t) &= x_2(t) &#h(5em) f_1(x,u,t) &= x_2  \
    dot(x)_2(t) &= -  frac(k(t),M)x_1(t) +  frac(1,M) u(t) & f_2(x,u,t) &= -  frac(k(t),M)x_1 +  frac(1,M)u  \
    y(t) &= x_1(t)
$
$f_2$ dipende esplicitamente da $t$ attraverso $k(t)$ quindi è un sistema tempo  #underline[variante]. Se invece $k(t) =  overline(k)$ (quindi una costante) per ogni $t$ allora il sistema è tempo #underline[invariante].  
Siccome $f_1$ e $f_2$ dipendono linearmente da $x$ e $u$ il sistema è  #underline[lineare].
$
    mat(delim: "[",
        dot(x)_1(t) ; 
        dot(x)_2(t)
    )
    &=
    underbrace(mat(delim: "[",
        0 , 1;  
        - frac(k(t),M) , 0
    ),A)
    mat(delim: "[",
        x_1(t);
        x_2(t)
    )
    +
    underbrace(mat(delim: "[",
        0  ;
         frac(1,M)
    ),B)
    u(t)
    \
    y(t) &= 
    underbrace(mat(delim: "[",
        1 , 0
    ),C)
    mat(delim: "[",
        x_1(t) ; 
        x_2(t)
    )
    +  underbrace(0,D) u(t)
$
per $k$ costante:
$
    A &= mat(delim: "[",
        0 , 1  ;
        -  frac(k,M) , 0
    )
    space
    B &= 
    mat(delim: "[",
        0;
        1
    )
    space
    C &= 
    mat(delim: "[",
        1 , 0
    )
$


===  Sistemi lineari tempo-invarianti SISO
I sistemi lineari tempo-invarianti single input single output (SISO) sono caratterizzati dalle matrici $A  in RR^(n  times n) , B  in RR^(n  times 1), C  in RR^(1  times n) , D  in RR^(1  times 1)$, ovvero $B$ è un vettore, $C$ è un vettore riga e $D$ è uno scalare.


== Principio di sovrapposizione degli effetti
Prendiamo un sistema lineare (anche tempo-variante)
$
    dot(x)(t) &= A(t)x(t) + B(t)u(t)  \
    y(t) &= C(t)x(t) + D(t)u(t)
$
- sia $(x_a (t), u_a (t))$ traiettoria con $x_a (t_0)$ = $x_(0a)$ \
- sia $(x_b (t), u_b (t))$ traiettoria con $x_b (t_0)$ = $x_(0b)$ \
Allora $forall  alpha,  beta  in RR$ dato lo stato iniziale $x_(a b,t_0) =  alpha x_(0a)+ beta x_(0b)$, si ha che
$
    (x_(a b)(t), u_(a b)(t)) = ( alpha x_(a)(t) +  beta x_b (t),  alpha u_a (t)+ beta u_b (t))
$
è traiettoria del sistema, ovvero applicando come ingresso $u_(a b)= alpha u_a(t) +  beta u_b(t)$ la traiettoria di stato è $x_(a b) (t) =  alpha x_a (t) +  beta x_b (t)$
$
    cases(reverse: #true,
        alpha x_(0a)(t)+ beta x_(0b) (t)  \
        alpha u_a (t) +  beta u_b (t)
    ) 
    ==>
    alpha x_a (t) +  beta x_b (t)
$ <sovrapposizione_effetti>
*IMPORTANTE:* non vale per i sistemi non lineari.

#heading(level: 3, numbering: none)[Dimostrazione]

Per dimostrarlo dobbiamo provare che soddisfa l'equazione differenziale. \
Siccome $(x_a (t), u_a (t))$ e $(x_b (t), u_b (t))$ sono traiettorie del sistema, esse soddisfano la relazione (@traiettoria_di_un_sistema)
$
    dot(x)_a = A(t) x_a (t) + B(t) u_a (t) \
    dot(x)_b = A(t) x_b (t) + B(t) u_b (t)
$
$
    frac(d,d t)x_(a b,t) &=  alpha dot(x)_a (t) +  beta dot(x)_b (t)  \
    &=  alpha(A(t)x_a (t) + B(t)u_a (t)) +  beta (A(t)x_b (t) + B(t)u_b (t))  \
    &= A(t) ( alpha x_a (t) +  beta x_b (t) ) + B(t) ( alpha u_a (t) + beta u_b (t) )
$
Per sistemi lineari sotto opportune ipotesi su $A(t)$ e $B(t)$ si può dimostrare che la soluzione è unica. \ 
Si dimostra lo stesso anche per l'uscita.


== Evoluzione libera e evoluzione forzata
Sia $x_ell (t), t >= t_0$ la traiettoria di stato ottenuta per $x_ell (t_0) = x_0$ e $u_ell (t) = 0, t >= t_0$. \
Sia $x_f (t), t >= t_0$ la traiettoria di stato ottenuta per $x_f (t_0) = 0$ e $u_f (t) = u(t), t >= t_0$.

Applicando il principio di sovrapposizione degli effetti si ha che, fissato lo stato iniziale $x(t_0) = x_0$ e applicando l'ingresso $u(t), t >= t_0$, la traiettoria di stato è data da
$
    x(t) =  underbrace(x_ ell(t), "evoluzione" \ "libera") + underbrace(x_f(t), "evoluzione" \ "forzata")
$<evoluzione_libera_forzata>
L'*evoluzione libera* è definita come $x_ ell (t)$ per $t  >= t_0$, tale che $x_ ell (t_0)=x_0$ e $u_l (t)=0$ per $t  >= t_0$, e uscita $y_ ell (t)=C(t)x_ ell (t)$. 

L'*evoluzione forzata* è definita come $x_f (t)$ per $t >= t_0$, tale che $x_f (t_0)=0$ e $u_l (t)=u(t)$ per $t  >= t_0$, e uscita $y_f (t)=C(t)x_f (t)+D(t)u(t)$.
   
*IMPORTANTE:* non vale per i sistemi non lineari.

== Traiettorie di un sistema LTI

===  Traiettorie di un sistema LTI: esempio scalare
Definiamo un sistema lineare tempo invariante (LTI) scalare con $x  in RR$, $u  in RR$, $y  in RR$
$
    dot(x)(t) &= a x(t) + b u(t) &space x(0) &= x_0  \
    y(t) &= c x(t) + d u(t) 
$
dall'analisi matematica possiamo scrivere il sistema come soluzione omogenea + soluzione particolare
$
    x(t) &= e^(a t)x_0 +  integral_0^t e^(a(t- tau))b u( tau) d  tau \
    y(t) &= c e^(a t)x_0 + c  integral_0^t e^(a(t- tau))b u( tau) d  tau + d u(t)
$
ricordiamo che la funzione esponenziale si può scrivere come
$
    e^(a t) = 1 + a t +  frac((a t)^2,2!) +  frac((a t)^3,3!) + ...
$


===  Traiettorie di un sistema LTI: caso generale
Definiamo un sistema lineare tempo invariante (LTI) $x in RR^n, u in RR^m, y in RR^p$
$
    dot(x)(t) &= A x(t) + B u(t) &space &x(0) = x_0  \
    y(t) &= C x(t) + D u(t)
$
$
    underbrace(x(t),RR^n) &=  underbrace(e^(A t),RR^(n  times n))  underbrace(x_0,RR^n) +  integral_0^t e^(A(t- tau))B u( tau) d  tau \
    y(t) &= C e^(a t)x_0 + c  integral_0^t e^(A(t- tau))B u( tau) d  tau + D u(t)
$
Ricordiamo che l'esponenziale di matrice si può scrivere come
$
    e^(A t) = I + A t +  frac((A t)^2,2!) +  frac((A t)^3,3!) + ...
$
$
    x(t) =  underbrace(e^(A t) x_0, "evoluzione" \ "libera") +  underbrace(integral_0^t e^(A(t- tau))B u( tau) d  tau, "evoluzione" \ "forzata")
$
$
    x_ ell (t) &= e^(A t)x_0 &space x_f (t) &=  integral_0^t e^(A(t- tau))B u( tau) d  tau
$ <traiettorie_LTI>


===  Esempio sistema non forzato
$
    dot(x)_1(t) =  lambda_1 x_1 (t) &wide dot(x)_2 (t) =  lambda_2 x_2(t)
$
$
    mat(delim: "[",
        dot(x)_1(t);
        dot(x)_2(t)
    )
    =
     underbrace(mat(delim: "[",
         lambda_1 , 0;
        0 , lambda_2;
    ),A)
    mat(delim: "[",
        x_1(t); 
        x_2(t)
    )
$

$A :=  Lambda$ matrice diagonale.
  
Il nostro è un sistema non forzato, quindi c'è solo l'evoluzione libera:
$
    x(t) = e^(Lambda t)x_0
$
$
    e^(Lambda t) &= 
    mat(delim: "[",
        1 , 0; 
        0, 1;
    )
    +
    mat(delim: "[",
        lambda_1 , 0;
        0 , lambda_2;
    )
    +
    mat(delim: "[",
        lambda_1 , 0  ;
        0 ,  lambda_2
    )^2  frac(t^2,2!) + ... \
    &=mat(delim: "[",
        1 , 0;
        0 , 1;
    )
    +
    mat(delim: "[",
         lambda_1 , 0 ; 
        0 ,  lambda_2;
    )
    +
    mat(delim: "[",
        frac(lambda_1^2 t^2,2!) , 0;  
        0 ,  frac( lambda_2^2 t^2,2!)
    ) + ... \
    e^(a t) = 1 + a t +  frac((a t)^2,2!) +  frac((a t)^3,3!) + ... ==> quad &=
    mat(delim: "[",
        e^( lambda_1 t) , 0;
        0 , e^( lambda_2 t);
    )
$
Quindi nel caso generale di $ Lambda  in RR^(n  times n)$
$
    e^( Lambda t) = 
    mat(delim: "[",
        e^( lambda_1 t) , 0 , ... , 0;  
        0 , e^( lambda_2 t) , ... , 0;  
        dots.v, dots.v, dots.down, dots.v;  
        0 , 0 , ... , e^(lambda_n t)
    )
$


===  Proprietà della matrice esponenziale <proprieta_matrice_esponenziale>

Esponenziale e cambio di base:
$
    e^(T A T^(-1)) = T e^(A t)T^(-1)
$ <matrice_esponenziale>
Data una matrice $A  in RR^(n  times n)$, esiste $J$ matrice diagonale a blocchi, chiamata _matrice di Jordan_, che è unica a meno di permutazioni dei blocchi, tale che
$
    A = T^(-1) J T
$ <matrice_Jordan>

con $T$ matrice invertibile (matrice del cambio base). Questa formula viene chiamata _forma di Jordan_.
 
La matrice di Jordan è fatta in questo modo
#cfigure("Images/Jordan_matrix.png", 35%)
con $ lambda_i$ autovalore di $A$.
   
Utilizzando questa forma riconduco il calcolo di $e^(A t)$ al calcolo di 
$
    e^(
        mat(delim: "[", 
            lambda , 1 , 0 , ... , 0;  
            0 , ... , ... , ... ,0  ;
            ... , ... , ... , ... , 0  ;
            ... , ... , ... , ... , 1  ;
            0 , ... , ... , 0 ,  lambda;
        )
        )
        = e^( lambda t)
        mat(delim: "[",
            1 , t ,  frac(t^2,2!) , ...,... ;
            0 , 1 , t ,  frac(t^2,2!) , ...   ;
            ... , ... , ... , dots.down , ...  ;
            0 , ... , ... , ... , 1;
        )
$
(IMPORTANTE:) tutti gli elementi di $e^(A t)$ sono del tipo
$
    t^q e^( lambda t)
$
con $q$ intero e $ lambda_i$ autovalori di A.


== Rappresentazioni equivalenti
Effettuiamo un cambio di base mediante una matrice $T$
$
    hat(x)(t) = T x(t)
$
ed essendo $T$ invertibile
$
    x(t) = T^(-1)  hat(x)(t)
$
Sostituendo nell'equazione della dinamica si ottiene
$
    #text(fill: purple)[$T dot$] underbrace(T^(-1)  dot( hat(x))(t),dot(x)(t)) &= A  underbrace(T^(-1)  hat(x)(t),x(t)) + B u(t) #text(fill: purple)[$dot T$]
$
$
    dot(hat(x))(t) &= T A T^(-1)  hat(x)(t) + T B u(t) \ 
    y(t) &= C T^(-1)  hat(x)(t) + D u(t)
$
Allora chiamo $hat(A) = T A T^(-1),  hat(B)=T B,  hat(C) = C T^(-1),  hat(D) = D$
$
    dot(hat(x))(t)  &=  hat(A)  hat(x)(t) +  hat(B) u(t) \ 
    y(t) &=  hat(C)  hat(x)(t) +  hat(D) u(t)
$
se $T$ è una matrice tale che
$
    J = T A T^(-1)
$
allora
$
    dot(hat(x)) = J  hat(x)(t) + T B u(t)
$

L'evoluzione libera vale
$
    hat(x)_ ell (t) = e^(J T) hat(x)_0
$




== Modi di un sistema lineare tempo invariante
Prendiamo un sistema lineare tempo invariante con $x  in RR^n, u  in RR^m, y  in RR^p$ e $x(0)=x_0$
$
    dot(x)(t) &= A x(t) + B u(t)  \
    y(t) &= C x(t) + D u(t)
$
Indichiamo con $ lambda_1,..., lambda_r$ gli $r  <= n$ autovalori (reali o complessi coniugati) distinti della matrice $A$, con molteplicità algebrica $n_1,...,n_r  >= 0$ tali che $ display(sum ^r_(i=1)) n_i = n$.  

Le componenti dell'evoluzione libera dello stato $x_ ell (t)$ si possono scrivere come
#grid(columns: (1fr, 1fr, 1fr), column-gutter: 1fr,
  [],
  math.equation(block: true ,numbering: none)[$ x_(ell, j) = sum_(i=1)^r sum_(q = 1)^h_i gamma_(j i q) t^(q-1)e^(lambda_i t) $],
  align(horizon)[$ j = 1,...,n $]
)

per opportuni valori di $h_i  <= n_i$, dove i coefficienti $ gamma_(j i q)$ dipendono dallo stato iniziale $x(0)$.  
I termini $t^(q-1)e^( lambda_i t)$  sono detti modi naturali del sistema. L'evoluzione libera dello stato è combinazione lineare dei modi.


=== Autovalori complessi  
Se la matrice $A$ è reale e $ lambda_i =  sigma_i + j  omega_i$ è un autovalore complesso, allora il suo complesso coniugato $ overline(lambda)_i =  sigma_i - j  omega_i$ è anch'esso autovalore di $A$.  
Inoltre si dimostra che i coefficienti $ gamma_(j i q)$ corrispondenti a $ lambda_i$ e $ overline(lambda)_i$ sono anch'essi complessi coniugati.  
Scriviamo allora l'*esponenziale di autovalori complessi coniugati*; se $ lambda_i =  sigma_i + j  omega_i$ e $ overline(lambda)_i =  sigma_i - j  omega_i$ allora
$
    e^( lambda_i t) &= e^( sigma_i + j  omega_i) &space e^( overline( lambda)_i t) &= e^( sigma_i - j  omega_i) \ 
    &= e^( sigma_i t) e^(j  omega_i t) & &= e^( sigma_i t) e^(-j  omega_i t) \
    &= e^( sigma_i t) ( cos( omega_i t) + j  sin( omega_i t)) & &= e^( sigma_i t) ( cos( omega_i t) - j  sin( omega_i t))
$
Si verifica quindi, per calcolo diretto, che le soluzioni $x_(ell,j)(t)$ sono sempre reali e che i modi del sistema corrispondenti ad autovalori complessi coniugati $ lambda_i$ e $ overline( lambda)_i$ sono del tipo
$
    t^(q-1) e^( sigma_i t)  cos ( omega_i t +  phi_i)
$
con opportuni valori della fase $ phi_i$.
   
Supponiamo che le molteplicità algebriche $n_1,...,n_r$ degli autovalori di $A$ coincidano cone le molteplicità geometriche (ad esempio quando gli autovalori sono distinti).  
Allora i coefficienti $h_i$ sono tutti pari a 1 e l'espressione dei modi si semplifica in 
$
    &e^( lambda_i t) &space &"per autovalori reali"  \
    &e^( sigma _i t)  cos ( omega_i t +  phi_i) & & "per autovalori complessi coniugati"
$<autovalori_complessi>

=== Modi naturali: autovalori reali semplici  <Modi_naturali_autovalori_reali_semplici>
#cfigure("Images/Autovalori_semplici.png", 70%)


=== Modi naturali: autovalori complessi coniugati semplici
#cfigure("Images/Autovalori_complessi.png", 70%)
#cfigure("Images/Esempio_autovalori.png", 70%)


=== Esempio sui modi naturali
$
    mat(delim: "[",
        dot(x)_1;  
        dot(x)_2
    )
    =
    mat(delim: "[",
        0 , 1;  
        a^2 , 0;
    )
    mat(delim: "[",
        x_1  ;
        x_2
    )
$

$
    p( lambda) &=  det ( lambda I -A) \
    &=  lambda^2 - a^2  \
    & => cases(
         lambda_1 = a\
         lambda_2 = -a
    )
$
I modi naturali di questo sistema sono 
$
    &e^(a t) &space space &e^(-a t)
$
Il modo $e^(a t)$ diverge a infinito, il che non è una cosa "buona" per dei sistemi di controllo, perché ad esempio se si sta realizzando un sistema di controllo della velocità vuol dire che la mia velocità sta aumentando, mentre dovrebbe rimanere fissa in un range.  
Non bisogna quindi focalizzarsi sul calcolare con precisione il valore dei modi naturali ma è importante conoscere come si comporta la loro parte reale.


=== Esempio 1
Consideriamo il seguente sistema LTI con $x  in RR^3$ e $u  in RR^3$
$
    dot(x)(t) =  underbrace(mat(delim: "[",
        0 , 1 , -1 ; 
        1 , -1 , -1 ; 
        2 , 1 , 3 
    ),A) x(t) +
     underbrace(mat(delim: "[",
        1 , 1 , 0  ;
        0 , 1 , 1  ;
        1 , 1 , 1
    ),B) u(t)
$
Mediante un cambio di coordinate usando la matrice $T = mat(delim: "[",
    0 , -1 , 1 ;  1 , 1 , -1  ; -1 , 0 , 1
)$ e ponendo $ hat(x)(t) = T x(T)$, il sistema si può riformulare come 
$
     hat(dot(x)) (t)=  underbrace(mat(delim: "[",
        -1 , 1 , 0;  
        0 , -1 , 0; 
        0 , 0 , -2 
    ), hat(A) = T A T^(-1))  hat(x)(t) +
     underbrace(mat(delim: "[",
        1 , 0 , 0;  
        0 , 1 , 0;  
        0 , 0 , 1
    ), hat(B) = T B) u(t)
$
Gli autovalori di $hat(A)$ sono $-1, -2$ con molteplicità algebrica $2,1$.
   
Per calcolare l'evoluzione libera consideriamo la formula vista in precedenza
$
     hat(x)_ ell  =e^(hat(A) t) hat(x)_0
$
Calcoliamo quindi l'esponenziale di matrice $e^( hat(A) t)$ per $ hat(A) = mat(delim: "[",
    -1 , 1 , 0;  
    0 , -1 , 0;  
    0 , 0 , -2 
)$
$
    e^( hat(A) t) &=  sum_(k=0)^ infinity mat(delim: "[",
        -1 , 1 , 0;  
        0 , -1 , 0;  
        0 , 0 , -2
    )^k  frac(t^k,k!) \
    &= mat(delim: "[",
        display(sum_(k=0)^ infinity) frac((-1)^k t^k,k!) , t  display(sum _(k=0)^ infinity)  frac((-1)^k t^k,k!) , 0;  
        0 ,  display(sum_(k=0)^ infinity)  frac((-1)^k t^k,k!) , 0;  
        0 , 0 ,  display(sum_(k=0)^ infinity)  frac((-2)^k t^k,k!)
    )  \
    &= mat(delim: "[",
        e^(-t) , t e^(-t) , 0;  
        0 , e^(-t) , 0;  
        0 , 0 , e^(-2t)
    )
$
quindi l'evoluzione libera dello stato è
$
    hat(x)_ ell = 
    mat(delim: "[",
        e^(-t) , t e^(-t) , 0;  
        0 , e^(-t) , 0;  
        0 , 0 , e^(-2t);
    )  hat(x)_0
$

- Se ad esempio la condizione iniziale è $ hat(x)_0 = mat(delim: "[", 1 ; 0 ; 0 )$, allora
$
    hat(x)_ ell = mat(delim: "[", e^(-t) ; 0 ; 0 )
$
Scriviamolo nello coordinate originali
$
    x_ell(t) =  underbrace(mat(delim: "[",
        1,1,0;  
        0,1,1;  
        1,1,1;
    ),T^(-1))  hat(x)_ ell (t) = 
    mat(delim: "[",
        e^(-t);  
        0;  
        e^(-t)
    )
$
- Se prendiamo come condizione iniziale $ hat(x)_0 = mat(delim: "[", 0 ; 1; 0 )$, allora
$
    hat(x)_ ell = mat(delim: "[", t e^(-t) ; e^(-t) ; 0 )
$
Scriviamolo nello coordinate originali
$
    x_ ell(t) =  underbrace(mat(delim: "[",
        1,1,0;  
        0,1,1;  
        1,1,1
    ) ,T^(-1))  hat(x)_ ell (t) = 
    mat(delim: "[",
        e^(-t) + t e^(-t);  
        e^(-t);  
        e^(-t) + t e^(-t)
    )
$


- Se prendiamo come condizione iniziale $ hat(x)_0 = mat(delim: "[", 0 ; 0 ; 1 )$. allora
$
     hat(x)_ ell = mat(delim: "[", 0 ; 0 ; e^(-2t) )
$
Nelle coordinate originali:
$
    x_ ell(t) =  underbrace(mat(delim: "[",
        1,1,0;  
        0,1,1;  
        1,1,1
    ) ,T^(-1))  hat(x)_ ell (t) = 
    mat(delim: "[",
        0;  
        e^(-2t);  
        e^(-2t)
    )
$




=== Esempio carrello
$
    dot(x)_1(t) &= x_2(t) \  
    dot(x)_2(t) &= -  frac(k(t),M)x_1(t) +  frac(1,M) u(t) \
    y(t) &= x_1(t)
$
$
    mat(delim: "[",
        dot(x)_1(t) ;
        dot(x)_2(t)
    ) &=
    mat(delim: "[",
        0 , 1  ;
        -  frac(k,M) , 0
    )
    mat(delim: "[",
        x_1(t);  
        x_2(t)
    )
    +
    mat(delim: "[",
        0;  
         frac(1,M)
    ) u(t)
    \
    y(t) &= mat(delim: "[",
        1 , 0
    )
    mat(delim: "[",
        x_1(t);  
        x_2(t)
    ) + 0 u(t)
$
Consideriamo $k$ costante, quindi sistema LTI.  
Gli autovalori della matrice $A$ sono $ lambda_1 = j  sqrt(dfrac(k,M)),  lambda_2 = -j  sqrt(dfrac(k,M))$ immaginari puri.
   
Applichiamo un controllo $u = - h x_2$. Le equazioni di stato del sistema diventano:
$
    dot(x)_1(t) &= x_2(t)\  
    dot(x)_2(t) &= - frac(k,M)x_1(t) -  frac(h,M)x_2(t)    
$
in forma matriciale
$
    mat(delim: "[",
        dot(x)_1(t);  
        dot(x)_2(t)
    ) &=
    mat(delim: "[",
        0 , 1;  
        -  frac(k,M) , - frac(h,M)
    )
    mat(delim: "[",
        x_1(t);  
        x_2(t)
    )
$
Quindi calcoliamo gli autovalori della matrice

//This command set all gaps in the matrices in the element with <big_matrices> label to 1em
//#show <big_matrices>: set math.mat(gap: 1em)

$
    
    A &= 
    mat(delim: "[", 
        0 , 1;  
        -  frac(k,M) , - frac(h,M)
    )
    &space space 
    A -  lambda I &= 
    mat(delim: "[",
        - lambda , 1;  
        -  frac(k,M) , - frac(h,M) -  lambda
    )
$

calcolando il determinante e ponendolo a zero si trova il polinomio caratteristico associato a essa
$
    & &space  lambda_1 &= -  frac(h,2M) +  sqrt( frac(h^2,4M^2) -  frac(k,M)) 
    \ 
    p( lambda) &=  lambda^2 +  lambda  frac(h,M) +  frac(k,M)  ==>
    \  
    & & lambda_2 &= -  frac(h,2M) -  sqrt( frac(h^2,4M^2) -  frac(k,M))
$

le cui soluzioni sono gli autovalori della matrice A.  
Prendiamo ora in considerazione la quantità sotto radice; è evidente che se $h^2 > 4 M k$ gli autovalori sono reali, mentre se $h^2 < 4 M k$ sono complessi coniugati.
   
Se invece $h^2 = 4 M k$, $ lambda_1 =  lambda_2 = - frac(h,2M)$, con molteplicità algebrica pari a 2. Si può dimostrare che la molteplicità geometrica è pari a 1, quindi il blocco di Jordan sarà $2  times 2$ (guardare  @proprieta_matrice_esponenziale)

//#show <big_matrices>: set math.mat(delim: "[", gap: 2em)

$
    J &= T A T^(-1) = 
    mat(delim: "[",
        -  frac(h,2M) , 1;  
        0 , -  frac(h,2M)
    )
    &space space 
    e^(J t) &= e^(-  frac(h,2M)t)
    mat(delim: "[",
        1 , t;  
        0 , 1
    )
$
$
    hat(x)_ ell = 
    mat(delim: "[",
        e^(-  frac(h,2M)t)  hat(x)_1(0) + t e^(-  frac(h,2M)t)  hat(x)_2(0);   
        e^(-  frac(h,2M)t)  hat(x)_2(0)
    )
    =
    mat(delim: "[",
        hat(x)_(1  ell)(t);  
        hat(x)_(2  ell)(t)
    )
$

Quindi i modi naturali del sistema sono
$
    &e^(-  frac(h,2M)t) &space space &t e^(-  frac(h,2M)t)  
$

da notare che anche si effettua il cambio di coordinate i modi del sistema non cambiano.

    - Supponiamo $ hat(x)(0) = mat(delim: "[", 1 ; 0 )$, allora
    $
        hat(x)_ ell (t) = 
        mat(delim: "[",
            e^(-  frac(h,2M)t)  hat(x)_1(0);
            0
        )
    $
    - Supponiamo $ hat(x)(0) = mat(delim: "[", 0 ; 1 )$, allora
    $
        hat(x)_ ell (t) = 
        mat(delim: "[",
            0 ;
            e^(-  frac(h,2M)t)  hat(x)_2(0)
        )
    $


#align(center)[
#cetz.canvas({
  import cetz.draw: *
  content((2, 4.2), [$y=t e^(- frac(h,2M)t)$], name: "text")
  plot.plot(
    size: (7,4), 
    x-tick-step: none, 
    y-tick-step: none, 
    axis-style: "left",
    plot-style: plot.palette.rainbow,
    {
    plot.add(
        domain: (0, 20), x => calc.sin(x * calc.pow(calc.e,-2/4 * x)))    
    }
    )
})]
    Si nota dal grafico che se $-  dfrac(h,2M)$ è "grande" il modo va a zero, quindi sono in un punto di equilibrio.

    - Se $h = 4M k = 0$ con $M >0,h=0,k=0$, il sistema diventa
    $
        mat(delim: "[",
            dot(x)_1(t);  
            dot(x)_2(t)
        ) =
        mat(delim: "[",
            0 , 1;  
            0 , 0;
        )
        mat(delim: "[",
            x_1(t);  
            x_2(t)
        )
    $
    i cui modi naturali sono $1,t$. È evidente che queste equazioni differenziali si possono scrivere come combinazione lineare dei modi:
    $
        x_1(t) &= x_1(0) + x_2(0)t\  
        x_2(t) &= x_2(0) 
    $


== Stabilità interna
=== Richiami sull'equilibrio di un sistema
Prendiamo un sistema lineare tempo invariante 
$
    dot(x)(t) = f(x(t), u(t))
$
Poniamo $u(t) = u_e    forall t  >= 0$, allora
#tageq($dot(x)(t) = f(x(t), u_e)$, $x(0)=x_0$)

Esiste, per un sistema di questo tipo, una $x_e$ tale che se $x(0)=x_e  ==> x(t) = x_e    forall t  >= 0$, quindi tale che se lo stato iniziale è costante la $x(t)$ rimane costante in ogni istante di tempo?

Chiamo $x_e$ equilibrio, $(x_e,u_e)$ la chiamo coppia stato-ingresso di equilibrio.
 
Proprietà fondamentale di una coppia di equilibrio è che
$
    f(x_e,u_e) = 0
$


=== Definizione generale
Per sistemi tempo-invarianti (anche se si può generalizzare) la _stabilità interna_ di un sistema è l'insieme delle conseguenze sulla traiettoria legate a incertezze sullo stato iniziale con ingressi fissi e noti.

 
=== Stabilità interna per sistemi non forzati <Stabilità_interna>
#tageq($dot(x)(t) = f(x(t))$, $x_e "equilibrio"$)

*Equilibrio stabile:* uno stato di equilibrio $x_e$ si dice stabile se $ forall  epsilon >0,  exists  delta >0$ tale che $ forall x_0 : || x_0-x_e ||  <=  delta$ allora risulti $ || x(t) - x_e || <  epsilon med forall t  >= 0$. 
 
*Equilibrio instabile:* uno stato di equilibrio $x_e$ si dice instabile se non è stabile.
   
*Equilibrio attrattivo:* uno stato di equilibrio $x_e$ si dice  attrattivo se $ exists  delta$ tale che $ forall x_0: || x_0-x_e ||  <=  delta$ allora risulti $ display(lim_(t  arrow  infinity)) || x(t)-x_e ||=0$; quindi se il sistema è in equilibrio solo a infinito.
   
*Equilibrio asintoticamente stabile:* uno stato di equilibrio $x_e$ si dice asintoticamente stabile se è stabile e attrattivo.
   
*Equilibrio marginalmente stabile:* uno stato di equilibrio si dice marginalmente stabile se è stabile ma non asintoticamente.
 
#figure(
    image("Images/Equilibrio_stabile.png", width: 55%),
    caption: [Rappresentazione grafica di un sistema in equilibrio stabile]
)

#figure(
    image("Images/Equilibrio_attrattivo.png", width: 55%),
    caption: [Rappresentazione grafica di un sistema in equilibrio attrattivo]
)

*N.B.* $delta < epsilon$.

=== Osservazioni
Le definizioni date sottintendono la parola locale, ovvero che la proprietà vale in un intorno dello stato di equilibrio $x_e$.
   
*Stabilità globale:* le proprietà di stabilità e asintotica stabilità sono globali se valgono per ogni $x  in RR^n$, invece che valere solo per $x_0$.
    
*Stabilità di una traiettoria:* le definizioni di stabilità si possono generalizzare a una traiettoria $ overline(x)(t), t  >= 0$.
#cfigure("Images/Stabilita_traiettoria.png", 70%)


== Stabilità interna di sistemi LTI
Nei sistemi lineari $x=0$ è sempre un equilibrio.
Per sistemi lineari si può dimostrare che tutti gli equilibri e tutte le traiettorie hanno le stesse proprietà di stabilità, tutte uguali a $x=0$. Per questo motivo si parla di *stabilità del sistema*.


#heading(level: 3, numbering: none)[Dimostrazione]
Sappiamo che
$
    dot(x)(t) =& A x(t) + B u(t) \
    A x_e +& B u_e = 0
    
$
allora supponiamo che  
$
    u(t) = u_e wide forall t  >= 0
$
Sia $ tilde(x)(t) := x(t) - x_e$, allora
$
     dot(tilde(x))(t) &= dot(x)(t) -  underbrace( frac(d,d t)x_e,0)  \
    &=A x(t)+B u_e \
    &=A( tilde(x)(t)+x_e)+B u_e  \
    &=A  tilde(x)(t) +  underbrace(A x_e+B u_e,0)
$
quindi
$
    dot(tilde(x))(t) = A  tilde(x)(t)
$

Concludiamo che
$
    dot(tilde(x))(t) = A  tilde(x)(t) = 0 <==>
    tilde(x)=0  <==>    x = x_e
$
cioè per studiare l'equilibrio di un sistema nel generico punto $x_e$ posso studiare l'equilibrio del sistema nell'origine.




=== Teorema  <Teorema_parte_reale_negativa>
Un sistema LTI è *asintoticamente stabile* #underline[se e solo se] tutti gli autovalori della matrice della dinamica hanno parte reale strettamente negativa.       

*N.B.* Se gli autovalori della matrice della dinamica hanno parte reale strettamente negativa i modi del sistema tendono a 0 (vedi @Modi_naturali_autovalori_reali_semplici[modi naturali di autovalori semplici])

=== Teorema
Un sistema LTI è stabile se e solo se tutti gli autovalori della matrice della dinamica hanno parte reale minore o uguale a zero e tutti gli autovalori a parte reale nulla hanno molteplicità geometrica uguale alla molteplicità algebrica (i mini blocchi di Jordan associati hanno dimensione uno).

#heading(level: 3, numbering: none)[Osservazione]
Si ha instabilità se almeno un autovalore della matrice della dinamica ha parte reale positiva o se almeno un autovalore con parte reale nulla ha molteplicità algebrica maggiore della molteplicità geometrica.

#heading(level: 3, numbering: none)[Osservazione]
La stabilità asintotica di sistemi LTI è sempre globale

#grid(columns: (1fr, 1fr, 1fr),
                column-gutter: 1fr,
                row-gutter: 4pt,
                [],
                math.equation(block: true ,numbering: none)[$x(0) &= x_0  ==> x(t)$], align(horizon)[$t  >= 0 $],
                [],
                math.equation(block: true ,numbering: none)[$x(0) &=  alpha x_0  ==>  alpha x(t)$], align(horizon)[$t  >= 0 $]
)



#heading(level: 3, numbering: none)[Proprietà]
Se un sistema LTI è globalmente asintoticamente stabile, $x=0$ è l'unico equilibrio. \ 
*Nota:* anche per sistemi #underline[non lineari] se $x_e$ è GAS (Globalmente Asintoticamente Stabile) allora è l'unico equilibrio.

//Questo perché RIVEDI



=== Esempio stabilità del sistema carrello
$
    dot(x)_1(t) &= x_2(t) \ 
    dot(x)_2(t) &= -  frac(k(t),M)x_1(t) +  frac(1,M) u(t)   
    y(t) &= x_1(t)
$
$
    mat(delim: "[",
        dot(x)_1(t);  
        dot(x)_2(t)
    ) &=
    mat(delim: "[",
        0 , 1;  
        -  frac(k,M) , 0
    )
    mat(delim: "[",
        x_1(t);  
        x_2(t)
    )
    +
    mat(delim: "[",
        0;  
         frac(1,M)
    ) u(t)
    \
    y(t) &= mat(delim: "[",
        1 , 0
    )
    mat(delim: "[",
        x_1(t);  
        x_2(t);
    ) + 0 u(t)
$
Consideriamo $k$ costante, quindi sistema LTI.  
Gli autovalori della matrice $A$ sono $ lambda_1 = j  sqrt( dfrac(k,M)),  lambda_2 = -j  sqrt( dfrac(k,M))$ immaginari puri, quindi sistema semplicemente (marginalmente) stabile.
  
Se applichiamo $u=-h x_2$ gli autovalori diventano $ lambda_1 = -  dfrac(h,2M) +  sqrt( dfrac(h^2,4M^2)- dfrac(k,M))$ e $ lambda_2 =  dfrac(h,2M) -  sqrt( dfrac(h^2,4M^2)- dfrac(k,M))$.  

    - Se $h^2  >= 4M k$ gli autovalori sono 2 reali negativi, quindi il sistema è asintoticamente stabile;
    - Se $h^2 < 4M k$ gli autovalori sono 2 complessi coniugati con parte reale negativa, quindi il sistema è asintoticamente stabile;
    - Se $h^2 = 4M k$, $ lambda_1 =  lambda_2 = - dfrac(h,2M)$, con molteplicità algebrica pari a 2. Si può dimostrare che la molteplicità geometrica è pari a 1, quindi il blocco di Jordan sarà $2  times 2$ (guardare le  @proprieta_matrice_esponenziale[proprietà della matrice esponenziale]) 
    $
        J &= T A T^(-1) = 
        mat(delim: "[",
            - frac(h,2M) , 1;  
            0 , -  frac(h,2M)
        )
        &space space 
        e^(J t) &= e^(-frac(h,2M)t)
        mat(delim: "[",
            1 , t;  
            0 , 1
        )
    $
    #h(0.8em)Gli autovalori sono a parte reale negativa, quindi il sistema è asintoticamente stabile;
    - Se $h=k=0  ==>  lambda_1= lambda_2=0$, quindi il sistema è instabile.





== Retroazione dello stato
Prendiamo un sistema lineare tempo invariante
$
    dot(x)(t) &= A x(t) + B u(t)\  
    y(t) &= C x(t) + D u(t) 
$
Supponendo di misurare l'intero stato, ovvero se $x(t)=y(t)$, allora possiamo progettare
$
    u(t) = K x(t) + v(t)
$
con $K  in RR^(m  times n)$ una matrice di guadagni e $v(t)$ un ulteriore ingresso per il sistema retroazionato
$
    dot(x)(t) = (A+B K)x(t)+B v(t)
$
Se vogliamo il sistema in anello chiuso asintoticamente stabile allora dobbiamo progettare $K$ tale che $(A + B K)$ abbia autovalori tutti a parte reale negativa.\  
*Nota:* la possibilità di scegliere gli autovalori di $(A + B K)$ (e.g., per renderli tutti a parte reale negativa) dipende dalla coppia $(A, B)$ ed è legata alla proprietà di *raggiungibilità*.

Se non è possibile misurare l'intero stato, ovvero se $y(t) != x(t)$, esistono tecniche per ricostruire lo stato a partire dalle misure mediante sistemi ausiliari chiamati *osservatori*.

Se sia possibile o meno ricostruire lo stato dipende dalla coppia $(A, C)$ ed è legato alla proprietà di *osservabilità*.

=== Proprietà di raggiungibilità (facoltativo)
Uno stato $tilde(x)$ si un sistema LTI si dice _raggiungibile_ se esistono un istante di tempo finito $tilde(t)>0$ e un ingresso $tilde(u)$, definito tra 0 e $tilde(t)$, tali che, detto $tilde(x)_f (t)$, $0<=t<=tilde(t)$, il movimento forzato dello stato generato da $tilde(u)$, risulti $tilde(x)_f (t) = tilde(x)$.

Un sistema i cui stati siano tutti raggiungibili si _completamente raggiungibile_.

Quindi, un particolare vettore $tilde(x)$ costituisce uno stato raggiungibile se è possibile, con un'opportuna scelta dell'ingresso, trasferire dall'origine al vettore in questione lo stato del sistema.

#heading(level: 3, numbering: none)[Teorema]
Detta $M_r$ _matrice di raggiungibilità_ definita come
$
    M_r = [B thin A B thin A^2 B thin dots thin A^(n-1) B ] in RR^(n times m n)
$
un sistema LTI è completamente raggiungibile, ovvero la coppia $(A,B)$ è completamente raggiungibile, se e solo se il rango della matrice di raggiungibilità è pari a $n$, cioè
$
    rho (M_r) = n
$


=== Proprietà di osservabilità (facoltativo)
Uno stato $tilde(x) != 0$ di un sistema LTI si dice _non osservabile_ se, qualunque sia $tilde(t) > 0$ finito, detto $tilde(y)_ell (t), t >= 0$, il movimento libero dell'uscita generato da $tilde(x)$ risulta $tilde(y)_ell (t) = 0, 0<=t<=tilde(t)$.

Un sistema privo di stati non osservabili si dice _completamente osservabile_.

Quindi, un particolare vettore $tilde(x)$ costituisce uno stato non osservabile se l'esame di un tratto di qualunque durata del movimento libero dell'uscita da esso generata non consente di distinguerlo dal vettore $x=0$.


#heading(level: 3, numbering: none)[Teorema]
Detta $M_o$ _matrice di osservabilità_, definita come
$
    M_o = [C^T thin A^T C^T thin A^T^2 C^T thin dots thin A^T^(n-1)C^T] in RR^(n times p n)
$
un sistema LTi è completamente osservabile, ovvero la coppia $(A,C)$ è completamente osservabile, se e solo se il rango della matrice di osservabilità è pari a $n$, cioè
$
    rho (M_o) = n
$



== Linearizzazione di sistemi in non lineari (tempo invarianti)
Prendiamo un  sistema non lineare tempo invariante
$
    dot(x)(t) &= f (x(t), u(t))\  
    y(t) &= h(x(t), u(t))
$
Sia $(x_e,u_e)$ una coppia di equilibrio, $f(x_e,u_e)=0$, consideriamo una traiettoria a partire da uno stato stato iniziale $x(0)=x_e+  tilde(x)_0$ 
$
    x(t) &= x_e +  tilde(x)(t)\  
    u(t) &= u_e +  tilde(u)(t)
$
con $y(t) = h(x_e , u_e ) +  tilde(y)(t) = y_e +  tilde(y)(t)$.

Essendo una traiettoria vale
$
    frac(d,d t)(x_e +  tilde(x)(t)) &= f (x_e +  tilde(x)(t), u_e +  tilde(u)(t))\  
    y_e +  tilde(y)(t) &= h(x_e+ tilde(x)(t), u_e+  tilde(u)(t))
$
Sviluppando in serie di Taylor (con $f$ e $h$ sufficientemente regolari) in $(x e , u e)$ #footnote[i termini del tipo $ frac(diff, diff x)f(x,u)$ vengono chiamati _Jacobiani_]
$
    frac(d,d t)(x_e +  tilde(x)(t)) &=  underbrace(f(x_e,u_e),0) +  underbrace(lr(frac(diff, diff x)f(x,u) |) _(x=x_e \ u=u_e),A_e) tilde(x)(t) +
    underbrace(lr(frac(diff, diff u)f(x,u)|)_(x=x_e \ u=u_e),B_e)  tilde(u)(t) +  "term. ord. sup." \
    y_e+ tilde(y)(t) &= h(x_e,u_e) +  underbrace(lr(frac( diff, diff x)h(x,u)|)_(x=x_e \ u=u_e),C_e) tilde(x)(t) +  underbrace(lr(frac(diff, diff u)h(x,u)|)_(x=x_e \ u=u_e),D_e) tilde(u)(t) +  "term. ord. sup."
$ 

$
    A &=  lr(frac(diff f(x,u), diff x)|)_(x=x_e \ u=u_e) 
    &space space 
    B &=  lr(frac( diff f(x,u), diff u)|)_(x=x_e \ u=u_e)
    \
    C &=  lr(frac( diff g(x,u), diff x)|)_(x=x_e \ u=u_e) 
    &
    D &=  lr(frac( diff g(x,u), diff u)|)_(x=x_e \ u=u_e)
$
quindi
#grid(columns: (1fr, 1fr, 1fr),
                column-gutter: 1fr,
                row-gutter: 4pt,
                [],
                math.equation(block: true ,numbering: none)[$tilde(dot(x))(t) &= A_e  tilde(x)(t) + B_e  tilde(u)(t) + "term. ord. sup."$],
                align(horizon)[$tilde(x)(0) =  tilde(x)_0$],
                [],
                math.equation(block: true ,numbering: none)[$tilde(y)(t) &= C_e  tilde(x)(t) + D_e tilde(u)(t) + "term. ord. sup."$],
                align(horizon)[]
)

Se consideriamo i termini di ordine superiore come un resto $ cal(R) (x,u)$ si osserva che
$
    lim_(||(tilde(x),tilde(u))|| arrow 0)  frac(||cal(R)( tilde(x), tilde(u))||, ||( tilde(x), tilde(u))||) = 0
$
di fatto è come se si avesse $ display(lim_(x  arrow 0)) dfrac(x^2,x)$.  
Quindi le due equazioni di prima si possono approssimare
$
     tilde(dot(x))(t) & approx A_e  tilde(x)(t) + B_e  tilde(u)(t)\  
     tilde(y)(t) & approx C_e  tilde(x)(t) + D_e tilde(u)(t) 
$
Il sistema linearizzato è
$
    Delta dot(x)(t) &= A_e  Delta x(t) + B_e  Delta u(t)\  
    Delta y(t) &= C_e  Delta x(t) + D_e  Delta u(t)
$
*N.B.* il pedice 'e' è una puntualizzazione ulteriore per sottolineare il fatto che le matrici sono valutate all'equilibrio, in altri testi potrebbero non avere questo pedice.

Le traiettorie del sistema non lineare soddisfano
$
    x(t) &= x_e +  tilde(x)(t)  &&approx x_e +  Delta x(t)\  
    u(t) &= u_e +  tilde(u)(t)  &&approx u_e +  Delta u(t)\  
    y(t) &= y_e +  Delta y(t)  &&approx y_e +  Delta y(t) 
$
per variazioni sufficientemente piccole.

*Nota:* $( Delta x(t), Delta u(t)), t  >= 0$ traiettoria del sistema linearizzato.



=== Esempio pendolo
$
    dot(x)_1(t) &= x_2(t) 
    &&= f_1(x(t),u(t))\  
    dot(x)_2(t) &= -frac(g,l) sin(x_1(t)) -frac(b,M  ell^2)x_2(t)+frac(1,M  ell^2) u(t) 
    &&= f_2(x(t),u(t))
$
$(x_e,u_e)$ coppia di equilibrio
$
    f(x_e,u_e)=0  arrow 
    cases(
        x_(2e)=0\  
        -  dfrac(g, ell)  sin(x_(1e)) -  dfrac(b,M  ell^2)x_(2e) +  dfrac(1,M  ell^2)u_e
    )
$
Prendiamo come equilibrio $x_e = mat(delim: "[", x_(1e); 0 )$, allora
$
    &-  dfrac(g, ell)  sin(x_(1e)) -  dfrac(b,M  ell^2)  dot 0 +  dfrac(1,M  ell^2)u_e = 0\  
     ==> & u_e = g M ell  sin(x_(1e))
$
Eseguiamo la linearizzazione intorno a $(x_e,u_e)$
$
     Delta dot(x)(t) = A_e  Delta x(t) + B_e  Delta u(t)
$

$
    underbrace(A_e, frac( diff f(x,u), diff x)|_(x=x_e \ u=u_e)) &=
    mat(delim: "[",
        dfrac(diff f_1(x,u), diff x_1) ,  dfrac( diff f_1(x,u), diff x_2);  
         dfrac( diff f_2(x,u), diff x_1) ,  dfrac( diff f_2(x,u), diff x_2);
    )_(x=x_e \ u=u_e)
    &space space 
    underbrace(B_e, frac(diff f(x,u), diff u)|_(x=x_e \ u=u_e)) &= 
    mat(delim: "[",
         dfrac(diff f_1(x,u), diff u);  
         dfrac(diff f_2(x,u),diff u)
    )
    \   
    &=
    mat(delim: "[",
        0 , 1;  
        - dfrac(g, ell) cos(x_1) , - dfrac(b,M  ell^2)
    )_(x=x_e \ u=u_e)
    &
    &= 
    mat(delim: "[",
        0;  
        dfrac(1,M  ell^2)
    )_(x=x_e \ u=u_e)
    \  
    &=
    mat(delim: "[",
        0 , 1;  
        - dfrac(g, ell) cos(x_(1e)) , - dfrac(b,M  ell^2)
    )
    &
    &=
    mat(delim: "[",
        0;  
        dfrac(1,M  ell^2)
    )
$


    - se $x_e = mat(delim: "[", 0 ; 0 )$ e $u_e = 0$
    $
        A_e &= 
        mat(delim: "[",
            0 , 1 ; 
            - dfrac(g, ell) , - dfrac(b,M  ell^2)
        ) 
        &space
        B_e &=
        mat(delim: "[",
            0;  
            dfrac(1,M  ell^2)
        )
    $
    - se $x_e=mat(delim: "[",  pi ; 0 )$ e $u_e=0$
    $
        A_e &= 
        mat(delim: "[",
            0 , 1;  
            dfrac(g, ell) , - dfrac(b,M  ell^2)
        ) 
        &space
        B_e &=
        mat(delim: "[",
            0;  
            dfrac(1,M  ell^2)
        )
    $
    - se $x_e=mat(delim: "[",  pi/2 ;  0 )$ e $u_e=M G ell$
    $
        A_e &= 
        mat(delim: "[",
            0 , 1;  
            0 , - dfrac(b,M  ell^2)
        ) 
        &space
        B_e &=
        mat(delim: "[",
            0;  
            dfrac(1,M  ell^2)
        )
    $




=== Stabilità di 3 sistemi lineari (linearizzazione intorno a 3 diversi equilibri)
    #v(2em)
    #enum()[
    Se $x_e = mat(delim: "[", 0 ; 0 )$ e $u_e = 0$
    $
        A_e &= mat(delim: "[", 
        0,1;
        -dfrac(g, ell), - dfrac(b,M  ell^2) 
        ) 
        &space 
        p( lambda) &=  lambda  (  lambda +  frac(b,M ell^2)) +  frac(g, ell)\  
        & & &= lambda^2 +  frac(b,M ell^2) lambda +  frac(g, ell)
    $
    $
        lambda_(1 slash 2) = -frac(b,2M  ell^2)  plus.minus  sqrt( ( frac(b,2M ell^2) ) -  frac(g, ell))
    $
    Abbiamo 2 autovalori a parte reale negativa, quindi il sistema linearizzato è _asintoticamente stabile globalmente_.
    #v(2em)
    ][
    Se $x_e=mat(delim: "[",  pi ; 0 )$ e $u_e=0$
    $
        A_e &= mat(delim: "[", 
        0,1;
        dfrac(g, ell), - dfrac(b,M  ell^2) 
        ) 
        &space 
        p( lambda) &=  lambda  (  lambda +  frac(b,M ell^2)  ) -  frac(g, ell)\  
        & & &= lambda^2 +  frac(b,M ell^2) lambda -  frac(g, ell)
    $
    $
         lambda_(1 slash 2) = -  frac(b,2M  ell^2)  plus.minus  sqrt( underbrace( ( frac(b,2M ell^2) ) +  frac(g, ell), >0))  ==>
        cases(
            lambda_(1) = -  dfrac(b,2M  ell^2) -  sqrt( ( dfrac(b,2M ell^2) ) +  dfrac(g, ell)) quad<0  
            \
            lambda_(2) = -  dfrac(b,2M  ell^2) +  sqrt( ( dfrac(b,2M ell^2) ) +  dfrac(g, ell)) quad>0
        )
    $
    Dato che abbiamo un autovalore a parte reale positiva il sistema è _instabile_.
    #v(2em)
    ][ 
    Se poniamo $x_e = mat(delim: "[",  pi/2  ; 0 )$ e $u_e = M g ell$
    $
        A_e &= 
        mat(delim: "[", 
            0, 1;
            0, - dfrac(b,M  ell^2) 
        ) 
        &space 
        p( lambda) &=  lambda  (  lambda +  frac(b, M ell^2)  )
    $
    $
         lambda_1 &= 0 &space space   lambda_2&=- frac(b,M ell^2)
    $
    Il sistema linearizzato è _stabile_, ma non asintoticamente, cioè marginalmente stabile (ricordando il @Teorema_parte_reale_negativa[Teorema])
    ]



== Stabilità e linearizzazione

=== Teorema

Dato un sistema non lineare tempo invariante, $dot(x)(t)=f(x(t),u(t))$, sia $x_e,u_e$ una coppia di equilibrio. Se il sistema linearizzato intorno a $(x_e,u_e)$ è asintoticamente stabile, allora l'equilibrio $x_e$, relativo all'ingresso $u_e$, è #underline[(localmente) asintoticamente stabile].

=== Teorema
Dato un sistema non lineare tempo invariante, $dot(x)(t)=f(x(t),u(t))$, sia $x_e,u_e$ una coppia di equilibrio. Se il sistema linearizzato intorno a $(x_e,u_e)$ ha almeno un autovalore a parte reale positiva, allora l'equilibrio $x_e$, relativo all'ingresso $u_e$, è #underline[instabile].


=== Controllo non lineare mediante linearizzazione 
Consideriamo il sistema non lineare
$
    dot(x)(t) = f(x(t),u(t))
$
Linearizzazione intorno all'equilibrio $(x_e,u_e)$
$
    Delta dot(x)(t) = A  Delta x(t) + B  Delta u(t)
$
Proviamo a portare $ Delta x(t)$ a 0, ovvero $x(t)$ a $x_e$ "in modo approssimato". Usando la retroazione dello stato $ Delta u(t) = K  Delta x(t) +  Delta v(t)$ otteniamo il seguente sistema in anello chiuso
$
     Delta dot(x)(t) = (A_e + B_e K) Delta x(t) + B_e  Delta v(t)
$
Così sono in grado di progettare la matrice $K$ in modo che $A_e + B_e K$ sia asintoticamente stabile. 

Grazie ai teoremi sulla linearizzazione, $x_e$ risulta un equilibrio (localmente) asintoticamente stabile per il sistema non lineare in anello chiuso (detto _retroazionato_).

Visto che $ Delta x(t)  approx x(t) - x_e$
$
    u(t) = u_e + K(x(t) - x_e) +  tilde(v)(t)  approx u_e + K  Delta x(t) +  tilde(v)(t)
$
Perciò la legge di controllo finale sarà
$
    u(t) = u_e + K(x(t)-x_e) +  tilde(v)(t)
$
#cfigure("Images/Controllo_retroazionato.png", 67%)

  




= Trasformata di Laplace
== Definizione
Data una funzione complessa $f$ di variabile reale $t$, $f: RR  arrow CC$ (anche se per noi tipicamente saranno funzioni $f: RR  arrow RR$), sia $s =  sigma + j  omega$ una variabile complessa ($ sigma$ parte reale, $ omega$ parte immaginaria); definiamo la _Trasformata di Laplace_ di $f(t)$
$
    F(s) =  integral_(0^-)^(+  infinity) f(t) e^(-s t) d t
$
se esiste per qualche $s$, ovvero se l'integrale converge.  
Includiamo nell'integrale $0^(-)$ per tener conto di eventuali impulsi cone la _delta di Dirac_.
 
*Notazione:* indichiamo la trasformata di Laplace con $ cal(L)$ tale che
$   
    f(t) xarrow(width: #3em, #text(size: 10pt)[$cal(L)$] ) F(s)
$ 



con $F: CC  arrow  CC$; indichiamo l'applicazione della trasformata con $F(s) =  cal(L)[f(t)]$.
 

== Osservazioni
=== Ascissa di convergenza
Sia $ overline( sigma)>-  infinity$ estremo inferiore di $s= sigma + j  omega$ per cui l'integrale converge. Allora la trasformata di Laplace esiste nel semipiano $ Re (s)> overline( sigma)$. \
$ overline( sigma)$ viene chiamata _ascissa di convergenza_.  
La trasformata di Laplace risulta essere una _funzione analitica_ e, grazie alle particolari proprietà delle funzioni analitiche, la sua definizione può essere estesa anche in punti di $s$ tali che $  Re(s) <=  overline(sigma)$, indipendentemente dal fatto che l'integrale non converga.
  
Dato che 
$
    e^(-s t) = e^(-  sigma t)e^(-j  omega t)
$
possiamo dire che $e^(- sigma t)$ ci aiuta a ottenere un integrale che converge.
#align(center)[
#cetz.canvas({
  import cetz.draw: *
  content((0.8, 3.5), [$e^(- sigma t)$], name: "text")
  plot.plot(
    size: (6,4), 
    x-tick-step: none, 
    y-tick-step: none, 
    axis-style: "left",
    plot-style: plot.palette.rainbow,
    {
    plot.add(
        domain: (0, 5), x => calc.pow(calc.e, -x))    
    }
    )
})]




=== Trasformate razionali <poli_e_zeri>
Di particolare importanza sono le _trasformate razionali_, cioè quelle in cui
$
    F(s) =  frac(N(s),D(s))
$<trasformate_razionali>

con $N(s)$ e $D(s)$ polinomi primi tra loro. Le radici di $N(s)=0$ si dicono *zeri* e quelle di $D(s)=0$ si dicono *poli*: nell'insieme, poli e zeri si dicono _singolarità_.
 

Se $f$ è reale allora i coefficienti dei polinomi $N(s)$ e $D(s)$ sono reali.

=== Esempio
$
    F(s) =  frac(s^2+2s, (s+1)(s+3)) =  frac(s(s+2), (s+1)(s+3))
$
allora
    - zeri di $F(s)$: $0$ e $-2$ 
    - poli di $F(s)$: $-1$ e $-3$



== Formula di antitrasformazione
La funzione trasformanda può essere ricavata dalla sua trasformata
mediante la _formula di antitrasformazione_
$
    f(t) =  frac(1, 2  pi j)  integral _( sigma-j  infinity)^( sigma+j  infinity) F(s) e^(s t) thin d s
$
*Notazione:* indichiamo l'antitrasformata di Laplace con $ cal(L)^(-1)$ tale che

#tageq($F(s) xarrow(width: #3em, #text(size: 10pt)[$cal(L)^(-1)$] )  f(t)$, $sigma >  overline( sigma)$)


indichiamo la formula di antitrasformazione con $f(t) =  cal(L)^(-1)[F(s)]$.
   
La $f(t)$ è fornita per $t >= 0$, perché solo nei punti di continuità in cui la $f$ è maggiore di zero essa contribuisce a determinare $F$. L'antitrasformata fornisce $f(t)=0$ per $t<0$, per questo la corrispondenza tra $f(t)$ e $F(s)$ è *biunivoca*.


=== Perché si utilizza la trasformata di Laplace
#cfigure("Images/Motivo_Laplace.png", 70%)

Se, provando a risolvere il problema oggetto, risulta difficile arrivare alla soluzione oggetto (magari perché i calcoli sono molto complessi o risulta poco conveniente in termini di risorse), allora si trasforma il problema oggetto in problema immagine con la trasformata di Laplace se risulta poi conveniente (o semplice) arrivare alla soluzione immagine, per poi antitrasformarla per ottenere la soluzione oggetto che si stava cercando.


== Proprietà della trasformata di Laplace
=== Linearità
Dati $f(t)$ e $g(t)$ tali per cui esistono le trasformate $F(s)$ e $G(s)$, allora $ forall  alpha  in  CC,  forall  beta  in  CC$ risulta
$
     cal(L)[ alpha f(t) +  beta g(t)] =  alpha  cal(L)[f(t)] +  beta  cal(L)[g(t)] =  alpha F(s) +  beta G(s)
$<linearità_Laplace>

#heading(numbering: none, level: 3)[Dimostrazione]
$
     cal(L)[alpha f(t) +  beta g(t)] &=  integral_(0^(-))^(+ infinity) ( alpha f(t) +  beta g(t) ) e^(-s t)   d t\  
    &=  alpha  underbrace( integral_(0^(-))^(+ infinity) f(t)e^(-s t)   d t,F(s)) +  beta  underbrace( integral_(0^(-))^(+ infinity) g(t)e^(-s t)   d t,G(s))\  
    &=  alpha F(s) +  beta G(s)
$


=== Traslazione temporale
$
cal(L)[f(t- tau)] = e^(- tau s)F(s) space forall  tau >0
$<traslazione_temporale_Laplace>

$ tau$ deve essere maggiore di 0, altrimenti la $f(t)$ sarebbe diversa da 0 per un tempo negativo.


#heading(numbering: none, level: 3)[Dimostrazione]
$
    cal(L)[f(t- tau)] &=  integral_(0^-)^(+  infinity) f(t -  tau)e^(-s t) thin d t\ 
    & =_(rho = t- tau)  integral_(- tau^-)^(+  infinity) f( rho)e^(-s( rho+tau)) thin  d rho  
$
siccome la $f(t)$ è nulla per $t<0$ posso riscrivere gli estremi di integrazione
$
     integral_(0^-)^(+  infinity) f( rho)e^(-s( rho+tau))   d rho
    &=  underbrace( integral_(0^-)^(+  infinity) f( rho)e^( rho)   d rho,F(s))  dot e^(-s tau)\  
    &= F(s) e^(-s tau)
$
come volevasi dimostrare
$
     cal(L)[f(t- tau)] = e^(- tau s)F(s)
$




=== Traslazione nel dominio della variabile complessa
$
    cal(L)[e^( alpha t )f(t)] = F(s - 
    alpha)
$<Traslazione_dominio_variabile__complessa_Laplace>

=== Dimostrazione
$
    cal(L)[e^( alpha t )f(t)] &=  integral_(0^-)^(+  infinity) f(t)e^( alpha t)  dot e^(-s t) thin d t
    \   
    &=  integral_(0^-)^(+  infinity) f(t)e^(-(s- alpha)t) thin d t
    \  
    &= F(s- alpha)
$


=== Derivazione nel dominio del tempo
$
     cal(L) [ frac(d, d t)f(t) ] = s F(s) - f(0)
$<Derivazione_nel_dominio_del_tempo>

Calcoliamo la trasformata della derivata seconda
$
     cal(L) [ frac(d^2,d t^2)f(t) ] &=
     cal(L) [ frac(d, d t) underbrace( [ frac(d, d t)f(t) ],g(t)) ]\  
    &=s G(s) - g(0)\  
    &=s G(s) - f'(0)\  
    &=s(s F(s)-f(0)) - f'(0)\  
    &= s^2 F(s) - s f(0) - f'(0)
$
Quindi possiamo definire la _derivata n-sima nel tempo_
$
     cal(L) [ frac(d^n, d t^n)f(t) ] = s^n F(s) -  sum_(i=1)^n s^(n-i)  lr(frac(d^(i-1), d t^(i-1))f(t)|)_(t=0)
$<derivata_n-sima_Laplace>
La proprietà ci dice che, se la funzione e le sue derivate si annullano in $t=0$, derivare nel dominio del tempo equivale a moltiplicare per $s$ nel dominio della variabile complessa; infatti $s$ viene chiamato _operatore di derivazione_.


=== Derivazione nel dominio della variabile complessa 
Supponiamo $F(s)$ derivabile per tutti gli $s$; allora risulta
$
    cal(L)[t f(t)] = -frac(d F(s), d s)
$<derivazione_dominio_variabile_complessa>
la quale è estendibile al caso della trasformata $t^n  dot f(t)$.

#heading(numbering: none, level: 3)[Dimostrazione]
Considerando che $t e^(-s t) = - dfrac(d, d s)e^(-s t)$
$
    cal(L)[t f(t)] &=  integral_(0^+)^(+ infinity) t f(t)e^(-s t) thin d t\  
    &= integral_(0^+)^(+ infinity) f(t)  underbrace(t e^(-s t),- frac(d, d s)e^(-s t)) thin d t\  
    &= integral_(0^+)^(+ infinity) f(t)  (- frac(d,d s)e^(-s t) ) thin d t\  
    &=- frac(d,d s)  underbrace( integral_(0^+)^(+ infinity) f(t) e^(-s t) thin  d t,F(s))\  
    &= -  frac(d F(s), d s)
$


=== Integrazione nel tempo
Supponiamo che la funzione $f(t)$ sia integrabile tra 0 e $+  infinity$. Allora
$
     cal(L)  [ integral_0^t f( tau)   d tau ] =  frac(1, s) F(s)
$

#par(leading: 1.2em)[La proprietà ci dice che integrare nel dominio del tempo equivale a dividere per $s$ nel dominio della variabile complessa; infatti $dfrac(1, s)$ viene chiamato _operatore di integrazione_.]



=== Convoluzione nel tempo
Date due funzioni $f_1$ e $f_2$, il loro _prodotto di convoluzione_ è
$
    f_1(t)  ast f_2(t) =  integral_(- infinity)^(+ infinity) f_1(t - tau)f_2(t)   d tau =  integral_(- infinity)^(+ infinity) f_1( eta)f_2( eta)   d eta = f_2(t- eta)  ast f_1(t)
$<prodotto_convoluzione>

e si trova
$
    cal(L)[f_1(t)  ast f_2(t)] = F_1(s)  dot F_2(s)
$<convoluzione_tempo_Laplace> 

=== Teorema del valore iniziale
Se una funzione reale $f(t)$ ha trasformata razionale $F(s)$ con grado del denominatore maggiore del grado del numeratore, allora
$
    f(0) =  lim_(s arrow  infinity) s F(s)
$<valore_valore_iniziale>

Se $f$ è una funzione discontinua di prima specie in $t=0$, $f(0)$ si interpreta come $f(0^+)$. L'equazione vale se $f(0)$ o $f(0^+)$ esistono.


=== Teorema del valore finale<teorema_valore_finale>
Se una funzione reale $f(t)$ ha trasformata razionale $F(s)$ con grado del denominatore maggiore del grado del numeratore e poli nulli o con parte reale negativa, allora
$
     lim_(t  arrow +  infinity) f(t) =  lim_(s arrow 0) s F(s)
$<teorema_valore_finale_eq>
L'equazione vale se $ display(lim_(t  arrow +  infinity)f(t))$ esiste.


== Trasformata di segnali elementari
Definiamo il _delta di Dirac_ $ delta(t)$ tale che
$
     integral_(0^-)^(0^+)  delta(t) thin d t = 1
$<delta_Dirac> 
#align(center)[
    #table(
    columns: (auto, auto),
    align: horizon,
    stroke: none, 
    $cal(L)[ delta(t)]=1$,
    image("Images/Delta.png", width: 60%),
    $ cal(L)[1(t)]= dfrac(1, s)$,
    image("Images/Scalino.png", width: 60%),
    $ cal(L)[t  dot 1(t)]= dfrac(1, s^2)$,
    image("Images/Scalino_2.png", width: 60%),
    $ cal(L)[e^( alpha t)  dot 1(t)]= dfrac(1,s- alpha)$,
    image("Images/Scalino_3.png", width: 60%),
    )
]




=== Trasformata della delta
$
    cal(L)[ delta(t)] &=  integral_(0^-)^(+ infinity)  delta(t) e^(-s t) thin  d t 
    \  
    &=  integral_(0^-)^0  delta(t)  underbrace(e^(-s dot 0),1)   d t +  underbrace( integral_(0^+)^(+ infinity)  underbrace( delta(t),0) e^(-s t) thin  d t,0)   
$

=== Trasformata del segnale gradino unitario
Il segnale gradino unitario $1(t)$ è definito
$
    1(t) = cases(
        0 &wide t<0\  
        1 &wide t >= 0
    )
$
$
     integral_(0^-)^(+  infinity) 1(t)e^(-s t)thin d t &=  integral_(0)^(+  infinity)  underbrace(1(t),1)e^(-s t)thin d t\  
    &=  integral_(0)^(+  infinity)e^(-s t) thin d t\  
    &=  lr(frac(e^(-s t), -s) |)_(t arrow+ infinity) -  lr(frac(e^(-0), -s) |)_(t=0)
$
$ lim_(t arrow+ infinity)e^(-s t)=0$, $e^0=1$
$
    underbrace(lr(size: #60%,frac(overbrace(e^(-s t), 0), -s) |)_(t arrow+ infinity), 0) 
    -lr(size: #60%, frac(overbrace(e^(-0), 1),-s) |)_(t=0) =  frac(1, s)
$

=== Trasformata del segnale rampa
Il segnale _rampa_ $r(t)$ è definito come
$
    r(t) = cases(
        0 &wide t<0\  
        t &wide t >= 0
    )
    space
    #image("Images/Segnale_rampa.png",width: 35%)
$ 
Per calcolare la trasformata del segnale rampa utilizziamo la proprietà di @eqt:derivazione_dominio_variabile_complessa[derivazione nel dominio della variabile complessa]
$
    cal(L)[t  dot 1(t)] &= - frac(d ( dfrac(1, s) ), d s)  \
    &=  frac(1, s^2)
$
   
Mentre per calcolare la trasformata del gradino moltiplicato un esponenziale utilizziamo la proprietà di @eqt:Traslazione_dominio_variabile__complessa_Laplace[di traslazione nel dominio della variabile complessa]. 
$
    cal(L) lr(size: #30%, [e^( alpha t)  underbrace(1(t),f(t))]) &=  underbrace(F(s- alpha),F(s)= frac(1, s)) \  
    &=  frac(1,s- alpha)
$

== Tabella delle trasformate <Tabella_trasformate>

#align(center)[

    #table(
        columns: (auto, auto),
        align: horizon,
        stroke: none,
        row-gutter: 10pt,
        $ cal(L)[ delta(t)]=1$,
        image("Images/Delta.png", width: 65%),
        $ cal(L)[1(t)]= dfrac(1,s)$,
        image("Images/Scalino.png", width: 65%),
        $ cal(L)[t  dot 1(t)]= dfrac(1,s^2)$,
        image("Images/Scalino_2.png", width: 65%),
        $ cal(L)[e^( alpha t)  dot 1(t)]= dfrac(1,s- alpha)$,
        image("Images/Scalino_3.png", width: 65%),
        $ cal(L)[ sin( omega t)1(t)]=  dfrac(omega, s^2+ omega^2)$,
        image("Images/Trasformata_seno.png", width: 65%),
        $ cal(L)[ cos( omega t)1(t)]=  dfrac(omega, s^2+ omega^2)$,
        image("Images/Trasformata_coseno.png", width: 65%),
        $ cal(L)[ sin( omega t +  phi)1(t)]=  dfrac( omega  cos phi plus.minus s  sin phi, s^2+ omega^2)$,
        [],
        $ cal(L)[ cos( omega t +  phi)1(t)]=  dfrac(s  cos phi  minus.plus  omega  sin phi, s^2+ omega^2)$
    )
]



= Funzione di trasferimento
== Introduzione
Consideriamo il sistema LTI con $x in RR^n, u in RR^m,y in RR^p$
$
    dot(x)(t) &= A x(t) + B u(t)\  
    y(t) &= C x(t) + D u(t)
$
con $x(0) = x_0$.

Siano $X(s):=  cal(L)[x(t)], U(s):=  cal(L)[u(t)]$ e $Y(s):=  cal(L)[y(t)]$. Applichiamo la trasformazione di Laplace ad ambo i membri delle equazioni precedenti, ricordando che $ cal(L) [ dfrac(d, d t)x(t) ]=s X(s)-x(0)$
$
    s X(s) - x(0) &= A X(s) + B U(s)\  
    Y(s) &= C X(s) + D U(s) 
$
se raccolgo $X(s)$ nella prima equazione
$
    (s I-A)X(s) =& x_0+ B U(s)  \
    Y(s) =& C X(s) + D U(s) 
$
$
    X(s) &= overbrace((s I-A)^(-1)x_0, X_ ell (s)) + overbrace((s I-A)^(-1) B U(s),X_f (s))  \
    Y(s) &= C X(s) + D U(s) 
$
Sottolineiamo che se avessimo un sistema generico non si potrebbe riscrivere come abbiamo fatto perché #underline[le matrici devono essere costanti].

Inoltre per poter scrivere un sistema LTI come sopra la matrice $(s I-A)$ deve essere invertibile; una matrice è invertibile se il suo determinante è non nullo, quindi, se $s$ è autovalore della matrice della dinamica e $p(s)$ è il polinomio caratteristico associato:
$
    p(s) =  det (s I-A)
$
Quindi le trasformate dello stato e dell'uscita del sistema in funzione di $x_0$ e $U(s)$ sono 
$
    X(s) &= overbrace((s I-A)^(-1)x_0, "evoluzione libera") + overbrace((s I-A)^(-1) B U(s),  "evoluzione forzata")\  
    Y(s) &= underbrace(C(s I-A)^(-1)x_0, "evoluzione libera") +  underbrace((C(s I-A)^(-1)B+D)U(s), "evoluzione forzata")
$
$
    X_ ell (s) &= (s I-A)^(-1)x_0 
    &space 
    X_f(s) &= (s I-A)^(-1) B U(s)
    \  
    Y_ ell (s) &= C(s I-A)^(-1)x_0 
    & 
    Y_f(s) &=  (C(s I-A)^(-1)B+D )U(s)  
$
Consideriamo ora la trasformata dell'evoluzione forzata dell'uscita
$
    Y_f (s) =  (C(s I-A)^(-1)B+D ) U(s)
$
la matrice
$
    G(s) = C(s I-A)^(-1)B+D
$
è detta _funzione di trasferimento_; se il sistema è SISO (Single Input Single Output) è una funzione scalare.
 
Abbiamo così ottenuto una *rappresentazione ingresso-uscita*
$
    Y_f (s) = G(s)U(s)
$<rappresentazione_ingresso_uscita>
se assumiamo che $x(0)=0$ otteniamo esattamente la trasformata di Laplace dell'uscita $y$
$
    Y(s) = G(s)U(s)
$<funzione_di_trasferimento>

Due osservazioni:
#list(
    marker: [#text(8pt)[$triangle.filled$]],
    [se si conosce la funzione di trasferimento $G(s)$ di un sistema e la trasformata di Laplace $U(s)$ dell'ingresso, è possibile calcolare, mediante antitrasformazione dell'equazione precedente @eqt:funzione_di_trasferimento[], il movimento forzato $y_f$ dell'uscita (che ovviamente coincide con il movimento $y$ se lo stato iniziale è nullo);],
    [la funzione di trasferimento è data dal rapporto tra la trasformata dell'uscita e dell'ingresso nel caso di $x(0)=0$
    $
        G(s) =  frac(Y(s),U(s))
    $]
)




== Richiami di calcolo matriciale

=== Matrice diagonale
Una _matrice diagonale_ è una matrice quadrata tale che per $i  != j$ si ha sempre $a_(i j)=0$ (ogni matrice diagonale è simmetrica).
$  
    mat(gap: #0.8em,
        1,0,0,0;
        0,7,0,0;
        0,0,9,0;
        0,0,0,-3
        ),
    mat(gap: #0.9em,
        0,0,0,0;
        0,2,0,0;
        0,0,3,0;
        0,0,0,1
        ) 
$


=== Matrice triangolare alta
Una _matrice triangolare alta_ è una matrice quadrata tale che per $i>j  a_(i j)=0$
$  
    mat(gap: #0.8em,
        a_(11),a_(12),a_(13),a_(14);
        a_(21),a_(22),a_(23),a_(24);
        a_(31),a_(32),a_(33),a_(34);
        a_(41),a_(42),a_(43),a_(44)  
        )  
        arrow  
    mat(gap: #0.8em,
        1,4,-3,7;
        0,6,-8,9;
        0,0,3,-5;
        0,0,0,1
        )   
$


=== Matrice triangolare bassa
Una _matrice triangolare alta_ è una matrice quadrata tale che per $i<j      a_(i j)=0$
$  mat(gap: #0.8em, 
    a_(11),a_(12),a_(13),a_(14);
    a_(21),a_(22),a_(23),a_(24);
    a_(31),a_(32),a_(33),a_(34);
    a_(41),a_(42),a_(43),a_(44);
    )  
    arrow  
    mat(gap: #0.8em,  
    1,0,0,0;
    6,7,0,0;
    3,-2,-9,0;
    5,4,-8,3;
    )   
$
Nota: una matrice diagonale è triangolare alta e triangolare bassa.


=== Matrice identità
$ 
    I_n =  
    mat(gap: #0.8em,
        1 , 0 , dots , 0 ; 
        0 , dots.down , dots , dots;   
        dots.v , dots.v , dots.down , dots.v;    
        0 , dots , dots , 1
    )
    space
    I_2 =  
    mat(gap: #0.8em,
         1 , 0;
        0 , 1;
        ) 
$


=== Trasposta di una matrice
$ mat(gap: #0.8em,
2 , 1 , 7;   
4 , 0 , 2
)^T =  
mat(gap: #0.8em,
    2 , 4;
    1 , 0;
    7 , 2
) 
$
$A = (a_(i j))$ significa che l'elemento di posto $(i,j)$ in $A$ è $a_(i,j)$.
  
$A^T := B = (b_(i j))$ con $b_(i j) = a_(j i)$ per ogni coppia di indici $(i,j)$.


=== Complemento algebrico
Definiamo $ hat(A)_(i j)$ complemento algebrico dell'elemento $a_(i j)$ il determinante della matrice ottenuta eliminando da $A$ la riga $i$ e la colonna $j$ (che chiamiamo $M$) e moltiplicando per $(-1)^(i+j)$
$
    hat(A)_(i j) = (-1)^(i+j)  det(M)
$


=== Determinante di una matrice
Il determinante di una matrice generica si calcola
$
    det(A) =  sum_(i=1)^n a_(i j)  hat(A)_(i j) =  sum_(j=1)^n a_(i j)  hat(A)_(i j)
$



== Funzione di trasferimento nel dettaglio
La funzione di trasferimento è definita
$
    G(s) = C(s I-A)^(-1)B + D
$
con $C$ matrice $1  times n$ e $B$ matrice $m  times 1$.
  
Definiamo ora la *matrice aggiunta* $ "adj"(A)$ come matrice dei complementi algebrici di $A$
$
    "adj"(A) = 
    mat(delim: "[",
        hat(A)_(11) ,  hat(A)_(12) ,  dots ,  hat(A)_(n 1);   
        hat(A)_(12) ,  hat(A)_(22) ,  dots ,  hat(A)_(n 2);   
        dots.v ,  dots.v  ,  dots.down ,  dots.v;  
        hat(A)_(n 1) ,  hat(A)_(n 2) ,  dots ,  hat(A)_(n n)
    )
$ 
La matrice inversa può essere definita con la matrice aggiunta:
$
    A^(-1) =  frac( "adj"(A), det(A))
$
Quindi, se consideriamo la nostra matrice $(s I-A)$
$
    (s I-A)^(-1) =  frac( "adj"(s I-A), underbrace( det(s I-A), "polinomio" \ "caratteristico di " A))
$
quindi scriviamo la matrice aggiunta di $(s I-A)$
$
    "adj"(s I-A) = 
    mat(delim: "[",
         hat((s I-A))_11 ,  hat((s I-A))_12 ,  dots ,   hat((s I-A))_(n 1);   
         hat((s I-A))_12 ,  hat((s I-A))_22 ,  dots ,  hat((s I-A))_(n 2);  
         dots.v ,  dots.v  ,  dots.down ,  dots.v;  
         hat((s I-A))_(n 1) ,  hat((s I-A))_(n 2) ,  dots ,  hat((s I-A))_(n n)
    )
$
matrice di polinomi in $s$ al più di grado $n-1$; il determinante di $s I-A$ è un polinomio in $s$ di grado $n$. Per cui 
$
    (s I-A)^(-1) =  underbrace(frac(1, det(s I-A)), "scalare")  dot  "adj"(s I-A)
$
Allora possiamo scrivere la funzione di trasferimento come
$
    G(s) =  frac(overbrace(N_(s p)(s), "polinomio di" \ "grado al più" n-1), underbrace(D_(s p)(s), "polinomio " \ "di grado" n))
    +  underbrace(D, "polinomio non " \ "strettamente" \ "proprio")
$
in forma estesa:
$
    G(s) =  frac(N(s),D(s)) =  frac(beta_( nu)s^( nu) +  beta_( nu-1)s^( nu-1) +  dots +  beta_1s +  beta_0,s^(nu #footnote[trasformata della delta di Dirac] )  +  alpha_( nu-1)s^( nu-1) +  dots +  alpha_1s +  alpha_0)
$

Le radici di $N(s)$ si dicono *zeri*, le radici di $D(s)$ si dicono *poli*; i poli sono radici di $ det(s I-A)$ quindi sono autovalori di $A$. Poli e zeri sono reali o complessi coniugati, poiché radici di polinomi a coefficienti reali.

=== Esempio 1
#par(leading: 1.2em)[Se prendiamo $y(t) =  dfrac(d,d t) u(t)$, allora la sua trasformata sarà $Y(s)=s U(s)$, quindi la funzione di trasferimento del sistema è $G(s)=s$; il sistema non è causale, perché il grado del numeratore ha grado maggiore di quello del denominatore.]
Questa considerazione diventa evidente se si utilizza la definizione di derivata:
$
    y(t) =  frac(d, d t)(u(t)) =  lim_(h arrow 0)  frac(u(t+h)-u(t),h)
$
infatti per conoscere la derivata in $t$ devo conoscere il valore del segnale in $t+h$.

=== Esempio 2
$
    y(t) &=  integral_(0)^t u(tau) d t 
    &space space
    Y(s) &=  frac(1,s)U(s)
$
in questo caso $G(s) = dfrac(1,s)$, quindi il grado del denominatore è maggiore di quello del numeratore, per questo il sistema è causale.


== Schema dell'utilizzo della trasformata di Laplace
#cfigure("Images/Schema_trasformata.png", 70%)



== Rappresentazioni e parametri della funzione di trasferimento
Può essere conveniente, in alcune situazioni, rappresentare la funzione di trasferimento in una delle seguenti forme fattorizzate
$ 
    G(s) =  frac(rho product_i (s + z_i) product_i (s^2+2  zeta_i  alpha_(n i)s  + alpha^2_(n i)),
                s^g  product_i (s + p_i) product_i (s^2+2  xi_i  omega_(n i)s +  omega^2_(n i)))
$<parametrizzazione_1>
$
    G(s) =  frac(mu  product_i (1 +  tau_i s) product_i (1 +  frac(2 zeta_i, alpha_(n i)) +  frac(s^2, alpha^2_(n i))),
                s^g  product_i (1 + T_i s) product_i (1 +  frac(2 xi_i, omega_(n i)) +  frac(s^2, omega^2_(n i))))
$<Forma_di_Bode>

*N.B.* il pedice $n$ sta per "naturale".
Con 

- lo scalare $ rho$ è detto costante di trasferimento, $ mu$ il _guadagno_;
- l'intero $g$ è detto _tipo_;
- gli scalari $-z_i$ e $-p_i$ sono gli zeri e i poli reali non nulli;
- gli scalari $ alpha_(n i) > 0$ e $ omega_(n i) > 0$ sono le _pulsazioni naturali_ delle coppie di zeri e poli complessi coniugati;
- gli scalari $ zeta_i$ e $ xi_i$, in modulo minori di 1, sono gli _smorzamenti_ degli zeri e dei poli complessi coniugati;
- gli scalari $ tau_i  != 0$ e $T_i  != 0$ sono le costanti di tempo

La seconda equazione @eqt:Forma_di_Bode[] è detta _forma di Bode_.

=== Esempio
$
    G(s) = 10  dot  frac(overbrace(s+10, product_i (s+z_i)),s^2(s+1)(s+100))
$
 
$
    z_1 &= 10 \ 
    p_1 &= 1 &space p_2 &= 100\  
     rho &= 10 & g &= 2
$
C'è un solo zero, che è $-10$, mentre i poli sono $0,-1,-100$.
   
Scriviamo la funzione di trasferimento nella seconda forma (di Bode):
$
    G(s) =  frac(cancel(10),s^2)  dot  frac(cancel(10)  dot  (1 +  dfrac(s,10) ), cancel(100)(1+s) (1 +  dfrac(s,100) )) = 1 dot  frac((1 + 0.1 s), s^2(1+s)(1+10^(-2)s))
$
$
    mu &= 1 &space space  tau_1 &= 0.1\  
    T_1 &= 1 & T_2 &= 10^(-2)
$
Prendiamo un polinomio di II grado del denominatore
$
    s^2 + 2  xi_i  omega_(n i)s +  omega^2_(n i)
$
le radici del polinomio sono:
$
    s_(1 slash 2) &= -  xi_i omega_(n i)  plus.minus  sqrt( xi_i^2 omega_(n i)^2 -  omega_(n i)^2)\  
    &=-  xi_i  omega_(n i)  plus.minus  omega_(n i)  sqrt( xi_i^2 - 1)
$
Se $|xi_i|<1$ abbiamo dei _poli complessi coniugati_
$
    s_(1 slash 2) = -  xi_i  omega_(n i) + j  omega_(n i)  sqrt(1 -  xi_i^2)
$
\
$
    |s_1| = |s_2| &=  sqrt( xi_i^2  omega_(n i)^2 +  omega_(n i)^2 (1 -  xi_i^2))\  
    &=  sqrt( xi_i^2  omega_(n i)^2 +  omega_(n i)^2  -  omega_(n i)^2 xi_i^2)  
    &=  omega_(n i)
$
Rappresentiamo i poli nel piano complesso:
#cfigure("Images/Poli_piano.png", 50%)


$ omega_(n i)$ quindi è il modulo delle radici, $ xi_i$ ci da invece informazioni sull'angolo delle radici nel piano complesso: se $ xi_i>0$ si hanno dei poli a parte reale negativa, se $ xi_i<0$ si hanno dei poli a parte reale positiva.  
Se $ xi_i=0$ gli autovalori sono immaginari puri, quindi i modi del sistema sono sinusoidi non smorzate, dato che lo smorzamento è nullo.
   
Nella rappresentazione classica si una una _x_ per i poli e un $circle$ per gli zeri
#cfigure("Images/Poli_rapp_classica.png", 70%)


== Cancellazioni
=== Esempio 1
$
    dot(x)_1 &= -x_1 + x_2\  
    dot(x)_2 &= -2x_2 + u\  
    y &= x_2
$
$
    G(s)    &= C(s I-A)^(-1) B\   
            &=  mat(delim: "[",
                0 , 1
                )
                mat(delim: "[",
                    s+1 , -1;  
                    0   , s+2 
                )^(-1)
                mat(delim: "[",
                    0 ,  1
                )\  
            &= mat(delim: "[",
                0 , 1
                )
                mat(delim: "[",
                     frac(s+2,(s+1)(s+2)) ,  frac(1,(s+1)(s+2)); 
                    0   ,  frac(s+2,(s+1)(s+2))
                )^(-1)
                mat(delim: "[",
                    0 ,  1
                )\
              
            &=  frac(cancel(s+1), cancel((s+1))(s+2))\  
            &=  frac(1,s+2)
$
Guardando questo esempio ci verrebbe da pensare che le cancellazioni sono innocue, questo perché stiamo cancellando il polinomio associato a un autovalore reale negativo, che quindi fa convergere il mio sistema. Cosa diversa è se cancelliamo un polinomio associato a un autovalore reale positivo.

=== Esempio 2
Infatti se prendiamo un sistema di questo tipo
$
    dot(x)_1 &= x_1 + x_2\  
    dot(x)_2 &= x_2 + u\ 
    y &= x_2
$
la funzione di trasferimento di questo sistema è 
$
    G(s) =  frac(cancel((s-1)), cancel((s-1))(s+2)) =  frac(1,s+2)
$
In questo caso stiamo cancellando un polinomio associato a un autovalore reale positivo, che quindi fa #underline[divergere] il sistema, perciò bisogna stare attenti quando si eseguono cancellazioni. Non basta guardare la funzione di trasferimento per conoscere l'andamento del sistema.


== Antitrasformazione di Laplace
Ricordiamo che la trasformata della risposta di un sistema Lineare Tempo Invariante (LTI) singolo ingresso singola uscita (SISO) è data da
$
    Y(s) = C(s I-A)^(-1) x(0) + G(s)U(s)
$
con $C(s I-A)^(-1)  in RR^(1  times n)$. Si può far vedere che gli elementi di $C(s I-A)^(-1)$ sono rapporti di polinomi.  

Nel corso della trattazione considereremo ingressi tali che $U (s)$ sia un rapporto di polinomi.
   
Quindi possiamo scrivere 
$
    Y(s) =  frac(N(s),D(s))
$
con $N (s)$ e $D(s)$ opportuni polinomi.
   
Ricordiamo che per $x(0)=0$ (risposta forzata)
$
    Y(s)=G(s)U(s)
$
Quindi, applicando in ingresso una delta di Dirac $u(t)= delta(t)$, che ha trasformata $U(s)=1$, si ha
$
    Y(s) = G(s)  
$
per questo per la risposta all'impulso le radici di $D(s)$ sono i poli di $G(s)$.





 = Sviluppo di Heaviside o in fratti semplici (poli distinti)

== Caso 1: poli reali o complessi coniugati distinti con molteplicità 1

Possiamo scrivere $Y(s)$ come 
$
    Y(s) =  frac(N(s),D(s)) =  frac(N(s), product_(i=1)^n (s + p_i)) =  sum_(i=1)^n  frac(k_i,s+p_i)
$
con $k_i$ detti residui. Consideriamo
$
    (s+p_i)  lr(frac(N(s),D(s))  |)_(s = -p_i) =  sum_( j=1  \ j != i)^n  lr(frac(k_j (s+p_i),s+p_j)  |)_(s=-p_i) + k_i
$
quindi ciascun residuo $k_i$ può essere calcolato come
$
    k_i = (s+p_i)  lr(frac(N(s),D(s))  |)_(s=-p_i)
$
*N.B.* $k_i$ reali se associati a poli reali, complessi coniugati se associati a una coppia di poli complessi coniugati.

Quindi, antitrasformando $Y(s)$ sviluppata in fratti semplici
$
    y(t) =  cal(L)^(-1)  [Y(s) ] =  sum_(i=1)^n k_i  cal(L) [ frac(1,s+p_i) ] =  sum_(i=1)^n k_i e^(-p_i t) 1(t)
$


=== Esempio
Vogliamo scrivere la $Y(s)$ in questo modo:
$
    Y(s) =  frac(s^2+s+1,(s+2)(s+10)(s+1)) =  frac(k_1,s+2) +  frac(k_2,s+10) +  frac(k_3,s+1)
$
allora
$
    lr((s+2)Y(s)  |)_(s=-2) &=  [frac( cancel((s+2))k_1, cancel(s+2)) +  frac(overbrace((s+2),0)k_2,s+10) +  frac(overbrace((s+2),0)k_3,s+1) ]_(s=-2)\  
    &= k_1
$
lo riscrivo con $Y(s)$ nella forma "originale"
$
    lr((s+2)Y(s)  |)_(s=-2) &= 
    lr(frac(cancel((s+2))(s^2+s+1), cancel((s+2))(s+10)(s+1))  |)_(s=-2)\  
    &= frac((-2)^2 + (-2)+1,(-2+10)(-2+1))   
    &= -  frac(3,8)
$
ergo
$
    k_1 = -  frac(3,8)
$
Calcoliamo anche le altre costanti
$
    lr((s+10)Y(s)  |)_(s=-10) &= 
    lr(frac(cancel((s+10))(s^2+s+1),(s+2) cancel((s+10))(s+1))  |)_(s=-10) \ 
    &= frac((-10)^2 + (-10)+1,(-10+2)(-10+1))\   
    &= frac(91,72) = k_2
$

$
    lr((s+1)Y(s) |)_(s=-1) &= 
    lr(frac(cancel((s+1))(s^2+s+1),(s+2)(s+10) cancel((s+1)))  |)_(s=-1) \ 
    &= frac((-1)^2 + (-1)+1,(-1+2)(-1+10))  \ 
    &= frac(1,9) = k_3
$
Quindi possiamo scrivere la $Y(s)$ come

$
Y(s) = -frac(3,8) underbrace(frac(1,s+2), #h(2em) arrow.b cal(L)^(-1) \ e^(-2t) 1(t)) + frac(91,72) underbrace(frac(1,s+10), #h(2em) arrow.b cal(L)^(-1) \ e^(-10t) 1(t)) + frac(1,9) underbrace(frac(1,s+1), #h(2em) arrow.b cal(L)^(-1) \ e^(-t) 1(t))
$

calcoliamo l'uscita del sistema con la formula di antitrasformazione
$
    y(t)    &=  cal(L)^(-1)  [Y(s)]\  
            &= -  frac(3,8)  cal(L)^(-1)  [ frac(1,s+2) ] +  frac(91,71)  cal(L)^(-1)  [ frac(1,s+10) ] +  frac(1,9)  cal(L)^(-1)  [ frac(1,s+1) ] \
            &=-  frac(3,8) e^(-2t)1(t) +  frac(91,71) e^(-10t)1(t) +  frac(1,9) e^(-t)1(t) 
$

#align(center)[
    #cetz.canvas({
        import cetz.draw: *
        content((3.1, 5.8), [$e^(-2t)$], name: "text")
        content((5.1,4.1), [$e^(-2t)1(t)$])
        plot.plot(
            size: (8,6), 
            x-tick-step: none, 
            y-tick-step: none,
            y-min: -2,
            y-max: 2, 
            x-min: -3,
            x-max: 3,
            axis-style: "school-book",
            {
            plot.add(
                domain: (0, 3), x => calc.pow(calc.e, -2*x),
                style: (stroke: black)
                )
            plot.add(
                domain: (-0.5, 0), x => calc.pow(calc.e, -2*x), style: (stroke: (dash: "dashed"))
            )   
            }
        )
    })
]


*N.B.* $1(t)$ definisce la funzione solo per $t  >= 0$.


=== Forma reale per poli complessi coniugati
Consideriamo la coppia di poli complessi coniugati
$
    p_(i,1) &=  sigma + j  omega &space space  p_(i,2) &=  sigma - j  omega
$
con residui associati (complessi coniugati)
$
    k_(i,1) &= M e^(-j phi) 
    &space space 
    k_(i,2) &= M e^(j phi)
$
L'antitrasformata dei due termini associati è data da (ricordando la @Tabella_trasformate[])
$
     cal(L)^(-1)  [frac(k_(i,1), s+p_(i,1)) +  frac(k_(i,2),s+p_(i,2)) ]      
                    &= M e^(-j  phi) e^(-p_(i,1)t)1(t) + M e^(j  phi) e^(-p_(i,2)t)1(t)\
                      
                    &= M e^(-j  phi) e^(-( sigma + j  omega)t)1(t) + M e^(j  phi) e^(-( sigma - j  omega)t)1(t)\
                      
                    &= 2M e^(- sigma t)  (e^(-j( omega t +  phi)) + e^(j( omega t +  phi)) )1(t)\
                      
                    &= 2M e^(- sigma t)  frac((e^(-j( omega t +  phi)) + e^(j( omega t +  phi)) ),2) 1(t)\
                      
                    frac(e^(j  alpha) + e^(-j  alpha),2) =  cos(alpha)  ==>
                    &= 2M e^(- sigma t)  cos( omega t +  phi) 1(t)
$

//#pagebreak()
=== Modi naturali di poli reali distinti
#cfigure("Images/Modi_naturali_poli_distinti_1.png", 70%)

=== Modi naturali di poli complessi coniugati distinti
#cfigure("Images/Modi_naturali_poli_distinti_2.png", 70%)

=== Modi naturali di un sistema LTI: poli distinti
#cfigure("Images/Modi_naturali_poli_distinti_3.png", 70%)



== Caso 2: Poli reali o complessi coniugati multipli con molteplicità maggiore di 1
$
    Y(s) =  frac(N(s),D(s)) =  frac(N(s), display(product_(i=1)^q) (s + p_i)^(n_i)) =  sum_(i=1)^q  sum_(h=1)^(n_i)  frac(k_(i,h),(s+p_i)^h)
$
con $k_(i,h), h=1, dots, n_i$ residui del poli $-p_i$. Consideriamo
$
    (s+p_i)^(n_i)  frac(N(s),D(s))
    &=(s+p_i)^(n_i)  sum_(j=1 \  j!= i)^q  sum_(h=1)^(n_j)  frac(k_(j,h),(s+p_j)^h) +  sum_(h=1)^(n_i) (s+p_i)^(n_i - h)k_(i,h)\
      
    &=(s+p_i)^(n_i)  sum_(j=1 \ j != i)^q  sum_(h=1)^(n_j)  frac(k_(j,h),(s+p_j)^h) +  sum_(h=1)^(n_i-1) (s+p_i)^(n_i - h)k_(i,h) + k_(i,n_i)
$
Quindi il residuo $k_(i,n_i)$ è dato da
$
    k_(i,n_i) = (s+p_i)^(n_i)  lr(frac(N(s),D(s)) |)_(s=-p_i)
$
Derivando $(s+p_i)^(n_i)  dfrac(N(s),D(s))$ si calcolano gli altri residui come
$
    k_(i,h) =  frac(1,(n_i-h)!)  frac(d^(n_i-h),d s^(n_i-h))  lr([(s+p_i)^(n_i)  frac(N(s),D(s)) ] |)_(s=-p_i)
$
Antitrasformando $Y(s)$ sviluppata in fratti semplici, ricordando la  @Tabella_trasformate[tabella delle trasformate] e la  proprietà di @eqt:derivazione_dominio_variabile_complessa[derivazione nel dominio della variabile complessa] 
$
    y(t) =  cal(L)^(-1) [Y(s)]
    &=  sum_(i=1)^q  sum_(h=1)^(n_i) k_(i,h)  cal(L)^(-1)  [frac(1,(s+p_i)^h) ]\
      
    &=  sum_(i=1)^q  sum_(h=1)^(n_i) k_(i,h)  frac(t^(h-1),(h-1)!)e^(-p_i t)1(t)
$



=== Esempio
Consideriamo la seguente trasformata di Laplace dell'uscita di un generico sistema
$
    Y(s) =  frac(s+3,(s+1)^2(s+2)) =  frac(k_(1,1),(s+1)) +  frac(k_(1,2),(s+1)^2) +  frac(k_(2),(s+2))   
$
$
    k_2 &= (s+2)Y(s) |_(s=-2) &space k_(1,2) = (s+1)^2 Y(s) |_(s=-1)
$
$
    k_2 
    &=  cancel((s+2))  lr(frac((s+3),(s+1)^2 cancel((s+2))) |)_(s=-2)
    &space
    k_(1,2)
    &=  cancel((s+1)^2)  lr(frac((s+3), cancel((s+1)^2)(s+2)) |)_(s=-1)\
      
    &=  frac((-2+3),(-2+1)^2)
    &
    &=  frac((-1+3),(-1+2))\
      
    &=1
    &
    &=2
$
$
    lr(frac(d, d s) ((s+1)^2 Y(s) ) |)_(s=-1) 
    &=  frac(d, d s)  [frac(k_(1,1)(s+1)^2,(s+1)) +  frac(k_(1,2) cancel((s+1)^2), cancel((s+1)^2)) +  frac(k_2(s+1)^2,s+2) ]_(s=-1)\
      
    &=  [k_(1,1) + 0 + k_2  frac(overbrace((s+1)^2,0)-2 overbrace((s+1),0)(s+2),(s+2)) ]_(s=-1)\
      
    &= k_(1,1)
$
$
    k_(1,1) 
    &=  frac(d, d s)  cancel((s+1)^2)  lr(frac(s+3, cancel((s+1)^2)(s+2)) |)_(s=-1)\
      
    &=  frac(d,d s) lr(frac(s+3,s+2) |)_(s=-1)\
       
    &=  lr(frac((s+2)-(s+3`),(s+2)^2) |)_(s=-1)\
      
    &= 1
$
quindi possiamo scrivere la $Y(s)$ come
$
    Y(s) =  frac(k_(1,1),(s+1)) +  frac(k_(1,2),(s+1)^2) +  frac(k_(2),(s+2)) =  frac(1,s+1) +  frac(2,(s+1)^2) +  frac(1,(s+2))
$
ricordando che $cal(L)^(-1) [dfrac(1,s^2)] = t 1(t)$ dalla @Tabella_trasformate[tabella delle trasformate] e che $cal(L)[e^(alpha t)f(t)] = F(s- alpha)$ dalla proprietà di @eqt:Traslazione_dominio_variabile__complessa_Laplace[traslazione nel dominio della variabile complessa], antitrasformiamo tutte le componenti della funzione di trasferimento
$
    cal(L)^(-1) [ frac(1,(s+1)) ] &= e^(-t) 1(t) 
    &space 
    cal(L)^(-1) [ frac(1,(s+2)) ] &= e^(-2t) 1(t) 
    &space 
    cal(L)^(-1) [ frac(1,(s+1)^2) ] &= e^(-t) t 1(t)
$
$
    y(t) 
    &= k_(1,1)e^(-t)1(t) + k_(1,2) t e^(-t)1(t) + k_(2)e^(-2t)1(t)\
      
    &= e^(-t)1(t) + 2 t e^(-t)1(t) +e^(-2t)1(t)
$



=== Forma reale per poli complessi coniugati con molteplicità maggiore di 1
Si può dimostrare che per una coppia di poli complessi coniugati
$
    sigma_i &+ j  omega_i  
    &space space 
    sigma_i &- j  omega_i 
$
con molteplicità $n_i$, il contributo elementare associato è dato da
$
    sum_(h=1)^(n_i) 2M_(i,h)  frac(t^(h-1),(h-1)!) e^(- sigma_i t)  cos( omega_i t +  sigma_(i,h)) 1(t)
$
Ad esempio, consideriamo la seguente funzione di trasferimento
$
    Y(s) =  frac(N(s),(s^2+2  xi  omega_n s +  omega_n^2))    
$
i poli della funzione sono
$
    & underbrace(- xi  omega_n, sigma) +  underbrace(j  omega_n  sqrt(1- xi^2), omega) 
    &space space  
    - xi  omega_n - j  omega_n  sqrt(1- xi^2)
$
quindi
$
    Y(s) =  frac(N(s),(s +  sigma + j  omega) (s +  sigma - j  omega))
$
e il contributo elementare 
$
    &2M_(1,1)e^(- sigma t) cos( omega t+ phi)1(t) + 2M_(1,2)t e^(- sigma t)  cos( omega t +  phi)1(t)
    \
    =& 2M_(1,1)e^(- xi  omega_n t)  cos( omega_n  sqrt(1 -  xi^2)t+ phi)1(t) + 2M_(1,2)e^(- xi  omega_n t)  cos( omega_n  sqrt(1 -  xi^2)t+ phi)1(t)
$


=== Modi naturali di poli multipli
Un modo naturale di un polo reale multiplo $-p_i$ è definito come
$
    frac(t^(h-1),(h-1)!) e^(-p_i t) 1(t)
$
I modi naturali di una coppia di poli complessi coniugati multipli $-( sigma_i + j  omega_i)$ e $-( sigma_i - j  omega_i)$ sono definiti come
$
    frac(t^(h-1),(h-1)!) e^(- sigma_i t)  cos( omega_i +  phi_(i,h)) 1(t)
$
#cfigure("Images/Modi_naturali_poli_multipli.png", 65%)




=== Modi naturali come risposta all'impulso
Sappiamo che per $x(0)=0$ (risposta forzata)
$
    Y(s) = G(s) U(s)
$
se applichiamo in ingresso una delta di Dirac $u(t)= delta(t)$
$
    Y(s) = G(s)    
$
Quindi la risposta ad un impulso è una combinazione lineare dei modi naturali del sistema lineare tempo invariante (SISO) descritto da $G(s)$.



== Risposta a un ingresso generico
Ricordiamo che 
$
    Y(s) =  underbrace(C(s I-A)^(-1), 
    #text(size: 6pt)[$dfrac(N_ ell (s),D(s))$] ) -x(0) +  underbrace(G(s)U(s), #text(size: 6pt)[$dfrac(N_f (s),D_f (s)) dfrac(N_u (s),D_u (s))$])
$


 
in cui $C(s I-A)^(-1) x(0)$, $G(s)$ e $U(s)$ sono rapporti di polinomi.  
Quindi
$
    y(t) 
    &= y_ ell(t) + y_f(t)\
      
    &=y_ ell(t) + y_(f,G)(t) + y_(f,U)(t)
$
in cui 

- $y_ ell (t)$ e $y_(f,G) (t)$ sono combinazioni lineari di modi naturali del sistema con matrici $A, B, C$ e $D$;
- $y_(f,U) (t)$ è combinazione lineare di "modi" presenti nell'ingresso $u(t)$ (dovuti alle radici del denominatore di $U (s)$).




== Risposta di sistemi elementari
Ricordiamo la  @eqt:parametrizzazione_1[formula di parametrizzazione]
$
    G(s) =  frac(rho  product_i (s + z_i) product_i (s^2+2  zeta_i  alpha_(n i)s  + alpha^2_(n i)),
    s^g  product_i (s + p_i) product_i (s^2+2  xi_i  omega_(n i)s +  omega^2_(n i)))
$

Consideriamo il caso di poli distinti. Da quanto visto fino ad ora risulta che per $x(0) = 0$ (risposta forzata)
$
    Y(s) = G(s)U(s) =  sum_(i)  frac(k_i,s+p_i) +  sum_i  frac(a_i s+b_i,s^2 + 2 xi_i omega_(n,i)s +  omega^2_(n,i))
$
#cfigure("Images/Risposta_sistemi_elementari.png", 60%)



== Stabilità esterna (BIBO) <BIBO>
Un sistema si dice BIBO (Bounded-Input Bounded-Output) stabile se la sua uscita forzata è limitata per ogni ingresso limitato.

Da quanto visto fino ad ora con lo sviluppo di Heaviside (fratti semplici) si può dedurre che un sistema con funzione di trasferimento $G(s)$ è BIBO stabile se e solo se tutti i poli di $G(s)$ sono a parte reale strettamente minore di zero.

*N.B.* La BIBO stabilità è equivalente alla stabilità asintotica. 




= Analisi di sistemi attraverso funzione di trasferimento
//in riferimento al pacchetto di slide "main_CAT_modulo2_part4"


== Dalla Funzione di Trasferimento allo spazio degli stati
Consideriamo la funzione di trasferimento
$
    G(s) =  frac( mu,1+T s)
$  

Questo tipo di sistemi può essere rappresentato nello spazio degli stati (la rappresentazione non è unica) come
$
    dot(x) &= -  frac(1,T) x +  frac( mu,T) mu\
    y &= x
$ <sis_1_ordine>
Infatti, la funzione di trasferimento associata alla @eqt:sis_1_ordine[] è 
$
    G(s) = C(s I-A)^(-1) B =  frac(mu,1+T s)
$
dove 
- il parametro $T$ è la costante di tempo associata al polo;
- il parametro $ mu$ è il _guadagno_



== Sistemi del primo ordine
Definiamo un sistema nello spazio della funzione di trasferimento
$
    G(s) &=  frac( mu,1+T s) &space U(s) =  frac(k,s)  
$
$
    Y(s) = G(s) U(s) =  frac( mu k,s(1+T s))
$
con $ mu >0,k>0,T>0$; da notare che se $T<0$ il sistema è instabile perché si avrebbe un polo positivo.

Allora mediante lo sviluppo di Heaviside e la formula di antitrasformazione troviamo che
$
    y(t) =  mu k(1-e^(-t slash T))1(t)
$
$y(0) = 0, thick dot(y)(0) =  dfrac( mu k,T) , thick y_( infinity) =  mu k$

#cfigure("Images/Sistemi_1_ordine_1.png", 50%)


Definiamo il *tempo di assestamento $ T_(a, epsilon)$* come il tempo tale per cui 

#tageq($(1-0.01 epsilon)y_( infinity)  <= y(t)  <= (1+0.01 epsilon)y_( infinity) $, $ forall t  >= T_(a, epsilon)$)

=== Esempio
Consideriamo un sistema con 
$
    G(s) =  frac( mu,1+T s) =  frac( mu,T)  frac(1,s+ frac(1,T))
$
con ingresso $u(t)=k 1(t)$, quindi $U(s) =  frac(k,s)$
$
    Y(s) 
    &= G(s) U(s)\
      
    &=   frac( mu,T)  frac(1,s+ frac(1,T))  frac(k,s)\
      
    &=  frac(k_1,s+ frac(1,T)) +  frac(k_2,s)
$
$
    y(t)
    &=  cal(L)^(-1)  underbrace( [ frac(k_1,s+ frac(1,T)) ],y_(G,t)) +  cal(L)^(-1)  underbrace( [ frac(k_2,s) ],y_(U,t))\
      
    &= k_1  underbrace(e^(-t slash T) 1(t), "sistema") + k_2  underbrace(1(t), "ingresso")
$
$
    k_1 
    &= lr((s+ frac(1,T))Y(s) |)_(s=-1/T)
    &space 
    k_2
    &= s Y(s) |_(s=0)\
      
    &=  cancel( (s+ frac(1,T) )) frac( mu,T) lr(frac(k, ( cancel(s+ frac(1,T)) )s) |)_(s=-1/T)
    &
    &= frac( mu,T)  lr(frac(k,s+ frac(1,T)) |)_(s=0)\
      
    &= frac( mu,T)  frac(k,- frac(1,T))
    &
    &=  mu k\

    &= - mu k
$
$
    y(t)
    &= -  mu k e^(-t/T)1(t) +  mu k 1(t)\
      
    &=  mu k (1 - e^(-t/T)) 1(t)
$
La risposta ottenuta ci dice che, dando in ingresso il gradino, il sistema ci metterà un po' per raggiungerlo, in base alla sua dinamica.

Per quanto riguarda invece il tempo di assestamento, esso si calcola
$
    T_(a, epsilon) = T  ln  ( frac(1,0.01 epsilon) )    
$

$
    T_(a,5) & approx 3T 
    &space space  
    T_(a,1) & approx 4.6T
$<tempo_assestamento>


=== Considerazioni

- Per calcolare la risposta riscrivere $G(s) =  dfrac( mu,T)  dfrac(1,s+ frac(1,T))$ e sviluppare $Y (s) = G(s)U (s)$ in fratti semplici;
- la risposta è monotona, i modi presenti sono $1(t)$ dell' ingresso e $e^(-t/T)$ del sistema;
- il valore asintotico è $mu k$, quindi se l'ingresso fosse un riferimento $k$ da seguire, avremmo un errore a regime $e_( infinity) = |1- mu|k$.





== Sistemi del secondo ordine
La funzione di trasferimento di sistemi del secondo ordine è 
$
    G(s) =  mu frac( omega^2_n,s^2+2  xi  omega_n +  omega^2_n)
$
Questo tipo di sistemi può essere rappresentato nello spazio degli stati come
$
    dot(x)_1 &= x_2\  
    dot(x)_2 &= -  omega^2_n - 2  xi  omega_n x_2 +  mu  omega^2_n u \ 
    y &= x_1
$
dove

- il parametro $ xi$ è il coefficiente di smorzamento;
- il paramento $ omega_n$ è la pulsazione naturale;
- il parametro $ mu$ è il guadagno.


=== Sistemi del secondo ordine con poli complessi coniugati
$
    G(s) &=  mu frac( omega^2_n,s^2+2  xi  omega_n +  omega^2_n) & U(s) &=  frac(k,s)\
      
    Y(s) &= G(s)U(s) =  mu k  frac( omega^2_n,s(s^2+2  xi  omega_n +  omega^2_n))
$
con $|xi|<1$ e $ omega_n > 0$
$
    y(t) =  mu k(1 - A e^(- xi  omega_n t) sin( omega t +  phi))1(t)
$
$
    A &=  frac(1, sqrt(1- xi^2)) 
    &space  
    omega &=  omega_n sqrt(1 -  xi^2) 
    &space  
    phi &=  arccos( xi)
$
$
    y(0) &= dot(y)(0) = 0 
    &space  
    dot.double(y)(0) &=  mu  omega^2_n 
    &space 
    y_( infinity) &=  mu k
$
$
    T_(a,5) & approx  frac(3, xi  omega_n) 
    &space space
    T_(a,1)  approx  frac(4.6, xi  omega_n)
$
Introduciamo un altro parametro che è la *sovraelongazione percentuale*, definita come
$
    S % = 100  frac(y_("max") - y_ infinity, y_( infinity))
$
con $y_( T(max))$ valore massimo e $y_( infinity)$ valore asintotico della risposta.

Per i sistemi del secondo ordine la sovraelongazione percentuale vale
$
    S % = 100 e^(- pi  xi slash sqrt(1- xi^2))
$
 
Analizziamo ora la risposta
$
    y(t) =  mu k(1 - A e^(- xi  omega_n t) sin( omega t +  phi))1(t)
$

#cfigure("Images/Sovraelongazione.png", 75%)

Dal grafico possiamo evincere che la sovraelongazione percentuale indica di quanto supero il valore stabile prima del transitorio. 

Come abbiamo visto prima, la sovraelongazione percentuale dipende solo dallo smorzamento, e, se scegliamo un valore massimo di sovraelongazione $S^star$, possiamo ricavare il valore di $ xi$ necessario:
$
    S %  <= S^star  <==>  xi  >=  frac( lr(|  ln  ( dfrac(S^star,100) )  |), sqrt( pi^2 +  ln^2  (  dfrac(S^star,100) )))
$


=== Luogo di punti a tempo di assestamento costante
Adesso proviamo a caratterizzare i sistemi del secondo ordine (con poli complessi coniugati) la cui risposta al gradino ha lo stesso tempo di assestamento.

Ricordiamo che 
- abbiamo approssimato $T_(a,5)  approx  frac(3, xi  omega_n)$ e $T_(a,1)  approx  frac(4.6, xi  omega_n)$
- $- xi  omega_n$ è la parte reale dei poli complessi coniugati

#cfigure("Images/Secondo_ordine_poli_complessi.png", 50%)


Quindi sistemi con poli complessi coniugati che hanno la stessa parte reale avranno una risposta al gradino con stesso tempo di assestamento.
   
Sul piano complesso i luoghi di punti a tempo di assestamento costante sono rette parallele all'asse immaginario.

#cfigure("Images/Tempo_assestamento_costante.png", 80%)

Nella figura vediamo in verde e in blu due coppie di poli distinti, ma con parte reale uguale; le risposte associate a questi poli sono evidentemente diverse (grafico in blu e grafico in verde), ma hanno lo stesso tempo di assestamento.


=== Luogo di punti a sovraelongazione costante
Proviamo a caratterizzare e i sistemi del secondo ordine (con poli complessi coniugati) la cui risposta al gradino ha la stessa sovraelongazione.

Ricordiamo che 
- $S % = 100 e^( frac(- pi  xi, sqrt(1- xi^2)))$
- $ arccos( xi)$ è l'angolo formato con l'asse reale

#cfigure("Images/Secondo_ordine_poli_complessi.png", 50%)

Quindi sistemi con stesso coefficiente di smorzamento $ xi$ avranno una risposta al gradino con stessa sovraelongazione.

Sul piano complesso i luoghi di punti a sovraelongazione costante sono semirette uscenti dall'origine.

#cfigure("Images/Sovraelongazione_costante.png", 80%)



=== Mappatura di specifiche temporali nel piano complesso
Vogliamo caratterizzare i sistemi del secondo ordine (con poli complessi coniugati) con $S %  <= S^star$ e $T_(a,5)  <= T^star$. 

Le specifiche sono soddisfatte per $ xi  >=  xi^star$ (con $ xi  <= 1$) e $ xi  omega_n  >=  dfrac(3,T^star)$ 

#cfigure("Images/Specifiche_temporali.png", 75%)

i poli complessi coniugati devono trovarsi nella zona colorata.


=== Sistemi del secondo ordine con poli reali
Caso $T_1  != T_2$ e $T_1 > T_2$
$
    G(s) &=  frac( mu,(1+T_1s)(1+T_2s)) & U(s) &=  frac(k,s)\  
    Y(s) &= G(s)U(s) =  frac( mu k,s(1+T_1s)(1+T_2s)) 
$
$
    & mu>0 &space &k>0 &space &T_1>0 &space &T_2>0
$

#cfigure("Images/Secondo_ordine_poli_reali_1.png", 37%)

$
    y(t) =  mu k  (1-  frac(T_1,T_1-T_2)e^(- frac(t,T_1)) +  frac(T_2,T_1-T_2)e^(- frac(t,T_2)) )1(t)
$
$
    &y(0) = 0 &space &dot(y)(0)=0 & &space dot.double(y)(0) =  frac( mu K,T_1 T_2) &space &y_( infinity)= mu k
$

#cfigure("Images/Secondo_ordine_poli_reali_2.png", 40%)



== Sistemi a polo dominante
Consideriamo $T_1 >> T_2$:

#align(center)[
    #cetz.canvas({
        import cetz.draw: *
        content((1.8, 3.7), [#text(red)[$e^(-t slash T_1) 1(t)$]])
        content((1.7, 1.7), [#text(green)[$e^(-t slash T_2)1(t)$]])
        plot.plot(
            size: (8,6), 
            x-tick-step: none, 
            y-tick-step: none,
            axis-style: "school-book",
            y-max: 1.5,
            {
            plot.add(
                domain: (0, 20), x => calc.pow(calc.e, -x/15),
                style: (stroke: red)
                )
            plot.add(
                domain: (0, 20), x => calc.pow(calc.e, -x/2), style: (stroke: green)
            )   
            }
        )
    })
]


Nella risposta $e^(- frac(t,T_2))$ tende a zero velocemente, quindi si può omettere, mentre $dfrac(T_2,T_1-T_2) << dfrac(T_1,T_1 - T_2)  approx 1$, quindi
$
    y(t)  approx  mu k (1-e^(- frac(t,T_1)) )1(t)
$

#cfigure("Images/Polo_dominante.png", 83%)


=== Esempio con coppia di poli complessi coniugati dominanti 
$
    G(s) =  mu  frac( omega_n^2,(s^2+2 xi omega_n s + omega_n^2)(s+p))
$
assumiamo $ dfrac(1,T)  >>  xi  omega_n$, cioè $- dfrac(1,T)  << -  xi  omega_n$

#cfigure("Images/Esempio_polo_dominante.png", 70%)

Prendiamo $U(s) =  dfrac(k,s)$
$
    Y(s) 
    &= G(s) U(s)\  
    &= frac( mu k  dfrac(omega_n^2,T),s(s^2+2 xi omega_n s +  omega_n^2) (s+ dfrac(1,T) ))\ 
    &=  frac(k_1,s) +  frac(k_2,s+ xi omega_n+j omega_n  sqrt(1- xi^2)) +  frac( overline(k_2),s+ xi omega_n-j omega_n  sqrt(1- xi^2)) +  frac(k_3,s+ dfrac(1,T))
$
Per $ dfrac(1,T)  >>  xi  omega_n$ la risposta si può approssimare a quella di sistemi del secondo ordine
$
    y(t)  approx  mu k  (1-A e^(- xi  omega_n t)  sin( omega t +  phi) )
$
Quindi, se ho più poli molto distanti dall'asse immaginario, posso sempre approssimare il sistema come un sistema del secondo ordine.
   
Infatti se analizziamo la risposta effettiva del sistema
$
    y(t) = k_1 1(t) + 2M e^(-  xi  omega_n t)  cos( omega_n  sqrt(1 -  xi^2)t +  phi_k) 1(t) + k_3 e^(-t/T) 1(t)
$
il termine $k_3 e^(-t/T) 1(t)$ va a 0 molto velocemente, quindi è trascurabile.



== Sistemi del secondo ordine con poli reali coincidenti
Caso $T_1=T_2$
$
    G(s) &=  frac( mu,(1+T_1s)^2) &space U(s) &=  frac(k,s)  \
    Y(s) &= G(s)U(s) =  frac( mu k,s(1+T_1 s)^2)
$
$ mu>0, k>0, T_1 >0$
$
    y(t) =  mu k  ( 1- e^(-t slash T_1) -  frac(t,T_1)e^(-t slash T_1)  ) 1(t)
$

#cfigure("Images/Poli_reali_coincidenti.png", 40%)

I modi presenti sono $1(t)$ (ingresso), $e^(-t slash (T-1))$ e $t e^(-t slash (T-1))$ (sistema).



== Sistemi del primo ordine con uno zero
$
    G(s) &=  mu  frac(1+  alpha T s,1+T s) 
    & space 
    U(s) &=  frac(k,s) \
    Y(s) &= G(s) U(s) =  mu k  frac(1+  alpha T s,s(1+T s))
$
$mu>0, thick k>0, thick T>0$

#cfigure("Images/Primo_ordine_uno_zero_1.png", 30%)

$
    y(t) &=  mu k (1+( alpha-1) e^(-t/T))1(t)\  
    y(0) &=  mu  alpha k 
    & space 
    y_( infinity) &=  mu k
$

#cfigure("Images/Primo_ordine_uno_zero_2.png", 60%)

Il grado relativo del sistema è zero, cioè il grado del numeratore è uguale al grado del denominatore; questo implica un collegamento algebrico tra ingresso e uscita, infatti $y(0)  != 0$; nello spazio degli stati questo equivale ad avere $D  != 0$.


== Sistemi del secondo ordine con poli reali e zero

=== Sistemi a fase non minima
$
    G(s) &=  mu frac(1 +  tau s,(1+T_1s)(1+T_2s)) 
    & space
    U(s) &=  frac(k,s) \ 
    Y(s) &= G(s)U(s) =  mu k  frac(1+  tau s,s(1+T_1s)(1+T_2s))
$
$ mu>0,k>0,T_1>0,T_2>0$
$
    y(t) &=  mu k  ( 1 -  frac(T_1- tau,T_1-T_2) e^(-t/T_1) +  frac(T_2- tau,T_1-T_2)e^(-t/T_2)  )1(t)  
    \
    y(0) &= 0 
    & wide  
    dot(y)(0) &=  frac( mu K  tau,T_1T_2) 
    & wide 
    y_( infinity) &=  mu k
$
*N.B.* il segno della derivata $dot(y)(0)$ dipende da $ tau$.
   
Nel caso in cui si abbia $T_1>T_2$ e $ tau<0$ il sistema è detto a *fase non minima*; in generale è detto sistema a *fase minima* un sistema caratterizzato da una funzione di trasferimento $G(s)$ con guadagno positivo, poli e zeri a parte reale negativa o nulla e non contenente ritardi (cioè se non ci sono esponenziali).

#figure(
    grid(
        columns: (50%, 50%),
        gutter: 2mm,
        image("Images/Poli_reali_zero_1.png", width: 70%),
        image("Images/Poli_reali_zero_2.png",width: 70%)
    )
)



la sottoelongazione $dot(y)(0)<0$ indica che il sistema inizialmente risponde in senso contrario rispetto all'ingresso.

Una bici, o una moto, che sterza è un sistema a fase non minima.
 


Dato un guadagno specifico, esiste  #underline[uno e un solo] sistema a fase minima che produce quel guadagno, mentre esistono infiniti sistemi a fase non minima che producono lo stesso. Per più informazioni: #link("https://youtu.be/jGEkmDRsq_M")[What Are Non-Minimum Phase Systems?].

#heading(level: 3, numbering: none)[Domanda]
Dato un sistema a fase non minima
#tageq($G(s) =  mu frac(1 +  tau s,(1+T_1 s)(1+T_2 s))$, $ tau < 0$)

posso progettare una $R(s) =  dfrac(1,1+  tau s)$ tale che
$
    R(s) G(s) =  mu  frac(1+  tau s,(1+T_1 s)(1+T_2 s)(1+  tau s))
$ 
così da eliminare lo zero positivo? \
Ovviamente no, perché $1+  tau s$ è un  modello della realtà, quindi, anche se matematicamente si può semplificare, nella realtà non si annullerà mai perfettamente. 


=== Sistemi a fase minima con sovraelongazione
Se $ tau>T_1>T_2$ abbiamo un sistema a *fase minima* con una sovraelongazione
$
    G(s) &=  mu frac(1 +  tau s,(1+T_1s)(1+T_2s)) 
    & space 
    U(s) &=  frac(k,s)  \
    Y(s) &= G(s)U(s) =  mu k  frac(1+  tau s,s(1+T_1s)(1+T_2s))
$
$ mu>0,k>0,T_1>0,T_2>0$

#cfigure("Images/Fase_minima_1.png", 40%)


$
    y(t) &=  mu k  ( 1 -  frac(T_1- tau,T_1-T_2) e^(-t slash (T-1)) +  frac(T_2- tau,T_1-T_2)e^(-t slash T_2)  )1(t)  
    \
    y(0) &= 0 & dot(y)(0) &=  frac( mu K  tau, T_1T_2) & y_( infinity) &=  mu k
$

#cfigure("Images/Fase_minima_2.png", 50%)


*Nota:* è presente una sovraelongazione tanto più accentuata quanto più lo zero è vicino all'origine (cioè al crescere di $ tau$).


=== Sistemi a fase minima con code di assestamento
Se  $ tau  approx T_1  >> T_2$ abbiamo un sistema a fase minima con code di assestamento.
$
    G(s) &=  mu frac(1 +  tau s,(1+T_1s)(1+T_2s)) 
    & space
    U(s) &=  frac(k,s) \
    Y(s) &= G(s)U(s) =  mu k  frac(1+  tau s,s(1+T_1s)(1+T_2s))
$
$ mu>0, thick k>0, thick T_1>0, thick T_2>0$

#cfigure("Images/Fase_minima_code_ass_1.png", 40%)

$
    y(t) &=  mu k  ( 1 -  frac(T_1- tau,T_1-T_2) e^(-t/T-1) +  frac(T_2- tau,T_1-T_2)e^(-t/T_2))1(t)\  
$
$
    y(0) &= 0 
    &wide  
    dot(y)(0) &=  frac( mu K  tau,T_1T_2) 
    &wide 
    y_( infinity) &=  mu k
$

#cfigure("Images/Fase_minima_code_ass_2.png", 40%)

A causa della non perfetta cancellazione polo/zero ($ tau  approx T_1$) il modo "lento" $e^(-t/T_1)$ è presente e il suo transitorio si esaurisce lentamente.








= Risposta in frequenza
== Risposta a un segnale di ingresso sinusoidale
Dato un sistema lineare tempo invariante SISO con funzione di trasferimento $G(s)$ vogliamo calcolare l'uscita in corrispondenza di un ingresso sinusoidale
$
    u(t) = U  cos( omega t +  phi)
$
la trasformata di Laplace di questo segnale è
$
    U(s) = U  frac(s  cos( phi) -  omega  sin( phi),s^2 +  omega^2)
$<trasformata_sinusoide>
quindi
$
    Y(s) = G(s) U(s) = G(s) U  frac(s cos( phi) -  omega  sin( phi),s^2 +  omega^2)
$<risposta_trasformata_sinusoide>

Consideriamo $G(s)$ con poli distinti a parte reale negativa (BIBO stabile). Sviluppando in fratti semplici si ha
$
    Y(s) =  sum_(i=1)^n  underbrace(frac(k_i,s+p_i),Y_1(s)) +  underbrace(frac(k_u,s-j  omega) +  frac(overline(k)_u,s+j  omega),Y_2(s))
$

Eseguiamo lo sviluppo di Heaviside sulla @eqt:trasformata_sinusoide[]:
$
    Y(s) = G(s) U(s) &= G(s) U  frac(s  cos( phi) -  omega  sin( phi),s^2 +  omega^2)\
      
    &= frac(N(s)U(s  cos( phi) -  omega  sin( phi)),(s+p_1)(s+p_2)  dots (s+p_n)(s^2 +  omega^2))\
      
    &=  frac(k_1,s+p_1) +  frac(k_2,s+p_2) +  dots +  frac(k_n,s+p_n) +  frac(k_u,s-j  omega) +  frac( overline(k)_u,s+j  omega)
$
quindi, antitrasformando
$
    y(t) &= k_1 e^(-p_1t)1(t) +  dots k_n e^(-p_n t)1(t) + 2 |k_u| underbrace(e^(-  sigma t), sigma = 0)  cos ( omega t +  arg(k_u))1(t)\
      
    &=  underbrace( sum_(i=1)^n k_i e^(-p_i t)1(t),y_1(t)) +  underbrace(2 |k_u|  cos ( omega t +  arg(k_u))1(t),y_2(t))
$
Poiché i poli di $G(s)$ sono a parte reale negativa, i contributi $e^(-p_i t)1(t)$ sono tutti convergenti a zero. Pertanto $y_1(t)  arrow 0$ per $t  arrow  infinity$.

Il residuo $k_u$ è dato da
$
    k_u &= (s - j  omega) Y(s) |_(s = j omega)\
      
    &= U G(s)  lr(frac(s  cos( phi) -  omega  sin( phi),s+j  omega) |)_(s = j omega)\
      
    &= U G(j omega)  frac(j omega  cos( phi) -  omega  sin( phi),j omega+j  omega)\
      
    &=U G(j omega)  frac(j  cos( phi) -  sin( phi),2j) \
      
    &= U G(j  omega)  frac( cos( phi) + j sin( phi),2)\
      
    #text(green)[$e^(j phi) =  cos( phi)+j  sin( phi)  ==>$] quad &= U G(j  omega)  frac(e^(j  phi),2)\
      
    &=  frac(U|G(j  omega)|,2) e^(j ( arg(G(j omega)) +  phi))
$
dove abbiamo scritto $G(j  omega) = |G(j  omega)| e^(j  arg(G(j omega)))$.

Ora che abbiamo calcolato $k_u$ possiamo sostituirlo nell' espressione di $y(t)$
$
    y(t) = y_1(t) + U |G(j  omega)|  cos( omega t +  phi +  arg(G(j omega)))
$
Siccome $y_1(t)  arrow 0$ per $t  arrow  infinity$, l'uscita $y(t)$ converge a 
$
    y_2(t) = U |G(j  omega)|  cos ( omega t +  phi +  arg(G(j omega)) )
$
ovvero, per $t$ sufficientemente grande si ha
$
    y(t)  approx U |G(j omega)|  cos( omega t +  phi +  arg(G(j  omega)))
$ <risposta_regime_permanente>
quest'espressione viene chiamata *risposta a regime  permanente*

#cfigure("Images/Risposta_regime_perm.png", 75%)


=== Teorema
Se a un sistema lineare tempo invariante con funzione di trasferimento $G(s)$ avente poli a parte reale negativa si applica l'ingresso sinusoidale
$
    u(t) = U  cos( omega t +  phi)
$
l'uscita, a transitorio esaurito, è data da
$
    y(t) = U |G(j omega)|  cos(omega t +  phi +  arg(G(j  omega)))
$


== Risposta a segnali periodici sviluppabili in serie di Fourier
Consideriamo un segnale d'ingresso $u(t)$ periodico, cioè $ exists T>0$ tale che $ forall t  >= 0 thick u(t+T) = u(t)$, che può essere sviluppato in serie di Fourier
$
    u(t) = U_0 + 2  sum_(n=1)^(+  infinity) |U_n|  cos  (n  omega_0 t +  arg(U_n) )
$
con $omega_0 =  dfrac(2  pi,T)$ e 
#tageq($ U_n =  frac(1,T)  integral_(t_0)^(t_0+T) u(t) e^(-j n  omega_0 t) d t$, $n=0,1,...$)


In base a quanto visto per un ingresso sinusoidale e sfruttando il principio di sovrapposizione degli effetti per sistemi BIBO stabili si può dimostrare che per $t$ elevati
$
    y(t)  approx Y_0 + 2  sum_(n=1)^(+  infinity) |Y_n|  cos(n  omega_0 t +  arg(Y_n))
$<risposta_svilupp_Fourier>
con $ omega_0 =  dfrac(2  pi,T)$ e 
#tageq($Y_n = G(j n omega_0)U_n$, $n=0,1,...$)


Il risultato appena visto può essere schematizzato come segue.  
Dato in ingresso un segnale periodico, esso può essere rappresentato come la somma delle armoniche dello sviluppo in serie di Fourier

#cfigure("Images/Fourier_1.png", 60%)

Sfruttando la sovrapposizione degli effetti tale schema è equivalente a considerare lo schema seguente per $t$ elevati
#cfigure("Images/Fourier_2.png", 60%)




== Risposta a segnali dotati di trasformata di Fourier
Dato un segnale non periodico dotato di trasformata di Fourier, possiamo scriverlo come
$
    u(t) =  frac(1,2 pi)  integral_(-  infinity)^(+ infinity) 2|U(j omega)|  cos( omega t +  arg(U(j omega))) thick  d omega
$
con 
$
    U(j omega) =  integral_(- infinity)^(+ infinity) u(t) e^(-j  omega t) thick d t
$
Ovvero l'ingresso è scomponibile come una infinità non numerabile di armoniche con valori di $ omega$ reali maggiori o uguali a zero.

Quindi se il sistema è BIBO stabile per $t$ elevati
$
    y(t)  approx  frac(1,2  pi)  integral_(-  infinity)^(+ infinity) 2|Y(j omega)|  cos( omega t +  arg(Y(j omega)))   d omega
$<risposta_trasformata_Fourier>

con 
$
    Y(j omega) = G(j  omega)U(j omega)    
$



== Richiami
=== Spettro di un segnale
Ogni segnale reale si può ottenere sommando opportune onde sinusoidali

#cfigure("Images/Spettro.png", 75%)

Uno stesso segnale può essere quindi visto equivalentemente nel dominio del tempo ($y(t)$) o delle frequenze ($Y(j omega)$). Le funzioni $y(t)$ e $Y(j omega)$ sono ugualmente informative e offrono due prospettive complementari per osservare lo stesso fenomeno.

La pulsazione $ omega$ e la frequenza $f$ sono legate dalla relazione $ omega = 2 pi f$.





== Definizione di Risposta in Frequenza
La funzione complessa $G(j omega)$ ottenuta valutando $G(s)$ per $s = j  omega$ è detta _risposta in frequenza_.
$
    G(s)  |_(s=j omega) = G(j omega)
$
La risposta in frequenza viene estesa anche a sistemi non asintoticamente stabili.

Per un certo valore di $ omega$, $G(j omega)$ è un numero complesso

#cfigure("Images/Risposta_in_frequenza_1.png", 60%)




=== Identificazione sperimentale della risposta in frequenza (approccio data-driven)
Nel caso in cui la risposta in frequenza non sia nota possiamo sfruttare i risultati precedenti per ricavarla sperimentalmente.

#cfigure("Images/Risposta_in_frequenza_2.png", 70%)

Diciamo che con questo tipo di approccio riusciamo a stimare la $G(j omega)$ (non la $G(s)$!).


== Rappresentazione della risposta in frequenza
Vi sono diversi modi di rappresentare la risposta in frequenza. Uno dei modi più usati sono i *diagrammi di Bode*, in cui si rappresentano separatamente $|G(j omega)|$ e $ arg(G(j omega))$ in funzione di $ omega$. Nei diagrammi di Bode si utilizza una scala logaritmica in base dieci per l'ascissa, dove è riportata la pulsazione $ omega$. In particolare, nei diagrammi di Bode si chiama _decade_ l'intervallo tra due pulsazioni che hanno un rapporto tra loro pari a dieci.  
Per come abbiamo definito i diagrammi, segue che la pulsazione nulla non compare nell'asse "finito" (si può avere pulsazione nulla solo a $- infinity$).
   
Nel tracciamento dei diagrammi di Bode conviene scrivere la funzione $G(s)$ nella forma fattorizzata @eqt:Forma_di_Bode[], qui riportata
$
    G(s) =  frac( mu  product_i (1 +  tau_i s) product_i (1 +  frac(2 zeta_i, alpha_(n i)) +  frac(s^2, alpha^2_(n i))),  s^g  product_i (1 + T_i s) product_i (1 +  frac(2 xi_i, omega_(n i)) +  frac(s^2, omega^2_(n i))))
$
e la risposta in frequenza associata corrispondente è
$
    G(j omega) =  frac(mu  product_i (1 + j omega tau_i) product_i (1 + 2j zeta_i frac( omega, alpha_(n i)) -  frac( omega^2, alpha^2_(n i))),
    (j omega)^g  product_i (1 + j omega T_i ) product_i (1 + 2j xi_i frac( omega, omega_(n i)) -  frac( omega^2, omega^2_(n i))))
$
Il diagramma delle ampiezze è espresso in *decibel*: $|G(j omega)|_("dB") = 20  log|G(j omega)|$.
Il diagramma delle fasi è espresso in gradi: $ arg(G(j omega))$


=== Proprietà dei numeri complessi e logaritmi
Dati due numeri complessi $a in  CC$ e $b  in  CC$ si ha
$
    |a b| = |a| |b|
$
e
$
    log(|a||b|) &=  log(|a|) +  log(|b|) 
    &space 
    arg(a b) &=  arg(a) +  arg(b) \ 
    log ( frac(|a|,|b|) ) &=  log(|a|) -  log(|b|) 
    &space   
    arg ( frac(a,b) ) &=  arg(a) -  arg(b)   \
    log(|a|^k) &= k log(|a|) 
    &space 
    arg(a^k) &= k arg(a)
$


=== Esempio operazioni in decibel
Definiamo il _modulo in decibel_ di un numero $a$
$
    |a|_("dB") = 20 log_(10)|a|
$

//log
#align(center)[
    #cetz.canvas({
        import cetz.draw: *
        content((1.2, 3.5), [$log(x)$], name: "text")
        plot.plot(
            size: (6.5,4.5),
            x-tick-step: none, 
            x-ticks: (1, 0), 
            y-tick-step: none,
            y-min: -2,
            y-max: 2, 
            x-min: 0,
            x-max: 20,
            axis-style: "school-book",
            {
            plot.add(
                domain: (0.1, 20), x => calc.log(x),
                style: (stroke: black)
                ) 
            }
        )
    })
]




inoltre
$
    |a|  >= 1 & ==> |a|_("dB")  >= 0  \
    0  <= |a|  <= 1 & ==> |a|_("dB")  <= 0
$


== Diagrammi di Bode
=== Diagramma del modulo
Partiamo col calcolare il valore del modulo della risposta in frequenza in Decibel
$
    |G(j omega)|_("dB") = 20  log_(10)|G(j omega)|
$
Prendiamo una risposta in frequenza così definita
$
    G(j omega) =  mu  frac((1+j omega tau)  (1 + 2  frac( zeta, alpha_n)j omega -  frac( omega^2, alpha_n^2) ),j omega(1+j omega T) (1 +  frac(2  xi, omega_n)j omega -  frac( omega^2, omega_n^2)  ))
$
$
    |G(j omega)|_("dB") =& 20  log|G(j omega)| \ 
    =&20 log|mu| + 20  log|1+j omega tau| + 20  log lr(|1+2  frac( zeta, alpha_n)j omega -  frac( omega^2, alpha_n^2) |)\
    & - 20  log|j omega| - 20  log|1+j omega T| - 20  log lr(|1+ frac(2 xi, omega_n)j omega -  frac( omega^2, omega_n^2) |)
    \  
    =&|mu|_("dB") + |1 + j omega tau|_("dB")  + abs(1+2  frac( zeta, alpha_n)j  omega -  frac( omega^2, alpha_n^2))_("dB") - |j omega|_("dB")\
    & - |1+j omega T|_("dB") -  abs(1+  frac(2 xi, omega_n)j omega -  frac( omega^2, omega_n^2))_("dB")
$


=== Diagramma della fase
$
    arg(G(j omega)) =&  arg  mu - g  arg(j omega) +  sum_(i)  arg(1+j omega tau_i) +  sum_i  arg (1 + 2 j  zeta_i  frac( omega, alpha_(n,1)) -  frac( omega^2, alpha^2_(n,i))   ) \
    &-  sum_(i)  arg(1+j omega T_i) -  sum_i  arg (1 + 2 j  xi_i  frac( omega, alpha_(n,1)) -  frac( omega^2, omega^2_(n,i))   )
$



=== Contributi elementari
Possiamo quindi studiare l'andamento dei seguenti contributi elementari
$
    G_a (j omega) &= mu 
    \ 
    G_b (j omega) &=  frac(1,(j omega)^g)  
    \
    G_c (j omega) &= (1+j omega tau_i) 
    & space  
    G_c (j omega) &= frac(1,1+j omega T_i)  
    \
    G_d (j omega) &=  ( 1+2j  zeta_i  frac( omega, alpha_(n,1) )-  frac( omega^2, alpha^2_(n,1))  ) & G_d (j omega) &=  frac(1, ( 1+2j  zeta_i  frac( omega, alpha_(n,1) )-  frac( omega^2, alpha^2_(n,1))  ))
$
La rappresentazione di questi diagrammi avviene su carte logaritmiche che vanno per _decade_, cioè per potenze di dieci.  



=== Guadagno statico
$
    G_a (j omega) &= mu 
    & space 
    |G_a (j omega)|_("dB") &= 20 log|mu| 
    & space   
    arg(G(j omega)) =  arg( mu)
$

#cfigure("Images/Diagramma_guadagno_statico_1.png", 63%)

Per quanto riguarda il diagramma dell'ampiezza,
- se $|mu|  >= 1$ allora $20 log|mu|   >= 0$
- se $|mu| <1$ allora $20 log|mu|  < 0$.  
Per il diagramma della fase invece
- se $ mu >0$ allora $ arg( mu)=0$
- se $ mu <0$ allora $ arg( mu)=-180 degree$.


=== Zeri nell'origine
Consideriamo una risposta con uno zero nell'origine (cioè $g=-1$)
$
    G_b (j omega) &=  frac(1,(j omega)^(-1)) = j omega 
    &wide 
    |G_b (j omega)|_("dB") &= 20  log  omega 
    &wide  
    arg(G_b (j  omega)) &=  arg(j omega)
$

#cfigure("Images/Diagramma_zero_origine.png", 72%)

La retta che definisce l'ampiezza $ log  omega arrow.r.bar 20  log  omega$ ha pendenza $20 "dB/dec"$; se ho $g$ zeri nell'origine allora la pendenza della retta sarà $20 dot g "dB/dec"$.\
$j omega$ è un punto sul semiasse immaginario positivo $forall omega > 0$, quindi fase $90^degree forall > 0$.


=== Poli nell'origine
Consideriamo una risposta con un polo nell'origine (cioè $g=1$)
$
    G_b (j omega) &=  frac(1,(j omega)^1) =  frac(1,j omega) 
    &space  
    |G_b (j omega)|_("dB") &= -20  log  omega 
    &space  
    arg(G_b (j  omega)) &= - arg(j omega)
$

#cfigure("Images/Diagramma_poli_origine.png", 75%)

Anche in questo caso, se ho $g$ poli nell'origine allora la pendenza della retta sarà $-20  "dB/dec"$.  
Per quanto riguarda la fase $-j  omega$ è un punto sul semiasse immaginario negativo $ forall  omega>0$, quindi la fase è $-90^ degree$.



=== Zero reale (ampiezza)
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

=== Zero reale negativo (fase)
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

#cfigure("Images/Diagramma_zero_reale_negativo_1.png", 60%)

se $ omega  >>  dfrac(1, tau)$ allora graficamente

#cfigure("Images/Diagramma_zero_reale_negativo_2.png", 60%)
#cfigure("Images/Diagramma_zero_reale_negativo_3.png", 72%)

il cambio di fase inizia circa una decade prima e finisce circa una decade dopo la pulsazione di taglio $ omega =  dfrac(1, tau)$.

#cfigure("Images/Diagramma_zero_reale_negativo_4.png", 60%)

$ dfrac(1,5) dot dfrac(1, tau) = 0.2 dot dfrac(1, tau) = 2  dot 10^(-1)  dfrac(1, tau)$, il doppio in scala logaritmica è $ dfrac(1,3)$ di una decade.


=== Zero reale positivo (fase)
Consideriamo $G_c (j omega) = 1 + j omega  tau,    tau <0$ (cioè una risposta con uno zero reale positivo)

#cfigure("Images/Diagramma_zero_reale_positivo_1.png", 32%)

$
     arg(G(j omega)) =  arg(1+j omega tau)  approx 
    cases(
        0 &  omega  <<  frac(1,|tau|) \  
        -90^( degree) wide&  omega  >>  frac(1,|tau|)
    )   
$

#cfigure("Images/Diagramma_zero_reale_positivo_2.png", 70%)



=== Polo reale
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

#cfigure("Images/Diagramma_polo_reale.png", 73%)

Il diagramma è uguale al diagramma dello zero ma ribaltato rispetto all'asse reale (consistentemente con il segno di $T$).



=== Polo reale negativo
Consideriamo $G_c (j omega) =  dfrac(1,1+j omega T),   T>0$ (cioè una risposta con un polo reale negativo)
$
    |G_c (j omega)|_("dB") &= -20  log  sqrt(1+ omega^2T^2) 
    & space  
    arg(G_c (j omega)) = - arg(1+j omega T)
$

#cfigure("Images/Diagramma_polo_reale_negativo_1.png", 53%)

Fino a $ dfrac(1,T)$, (pulsazione di taglio), si ha un andamento costante a $0 "dB" $, cioè il modulo della sinusoide in uscita non cambia. A partire da $ dfrac(1,T)$ si ha una retta $ log omega  arrow.r.bar 20  log  dfrac(1,|T|) - 20  log  omega$ con pendenza $-20   "dB/dec"$.  
    
Lo scostamento massimo (tra diagramma asintotico e diagramma reale) si ha in $ omega =  dfrac(1,T)$ dove
$
    |G_c (j omega)|_("dB") &= -20  log  sqrt(1+1) \ 
    &= -20  log  sqrt(2)  approx -3
$
Il cambio di fase inizia circa una decade prima e finisce circa una
decade dopo la pulsazione di taglio $ omega =  dfrac(1,T)$. 



=== Zeri complessi coniugati (ampiezza)
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

#cfigure("Images/Diagramma_zeri_cc_ampiezza_1.png", 55%)
#cfigure("Images/Diagramma_zeri_cc_ampiezza_2.png", 55%)

Il minimo dell'ampiezza si ha alla pulsazione $ omega_r =  alpha_n  sqrt(1-2 zeta^2)$ con $|G_d (j omega_r)| = 2 |zeta| sqrt(1- zeta^2)$


=== Zeri complessi coniugati a parte reale negativa (fase)
Consideriamo $G_d (j omega) = 1+ 2j  zeta  dfrac( omega, alpha_n) -  dfrac( omega^2, alpha_n^2),    zeta>0$, una risposta con una coppia di zeri complessi coniugati a parte reale negativa
$
     arg(G_d (j  omega))  approx 
    cases(
        0 &  omega  <<  alpha_n  \
        180^ degree wide& omega  >>  alpha_n
    )    
$

#cfigure("Images/Diagramma_zeri_cc_neg_1.png", 60%)

Vediamo che la risposta, per $ omega  >>  alpha_n$, nel piano complesso ha sicuramente una parte reale molto negativa, mentre la parte immaginaria dipende dal valore $ zeta$, il quale influenza molto l'andamento della fase. Ad esempio se $ zeta  arrow 0$ la parte immaginaria nel piano complesso tenderà anch'essa a 0, così da rendere molto facile appurare che l'argomento della nostra risposta sia quasi $180^ degree$. Anche con $ zeta =1$ l'argomento sarà circa $180^ degree$, ma solo per pulsazioni molto grandi, perché la parte reale tende a $ dfrac( omega^2, alpha_n^2)$, che è $O( dfrac( omega, alpha_n))$ (la parte immaginaria) per $ omega  arrow  infinity$.

#cfigure("Images/Diagramma_zeri_cc_neg_2.png", 73%)

Nel diagramma di fase più $ zeta$ è piccolo e più la discontinuità da $0^ degree$ a $180^ degree$ è rapida. 



=== Zeri complessi coniugati a parte reale positiva
Consideriamo $G_d (j omega) = 1+ 2j  zeta  dfrac( omega, alpha_n) -  dfrac( omega^2, alpha_n^2),  quad  zeta < 0$, una risposta con una coppia di zeri complessi coniugati a parte reale positiva.  
Il diagramma di fase di è speculare a quello precedente

#cfigure("Images/Diagramma_zeri_cc_pos_1.png", 73%)




=== Poli complessi coniugati a parte reale negativa <poli_complessi_coniugati_parte_reale_negativa>
Consideriamo una risposta in frequenza con poli complessi coniugati a parte reale negativa
$G_d (j omega) =  dfrac(1,1+2j  xi  frac( omega, omega_n)- frac( omega^2, omega_n^2)),    xi > 0$

#cfigure("Images/Diagramma_poli_cc_neg_1.png", 63%)

I diagrammi sono quelli precedenti ribaltati rispetto all'asse reale, infatti la retta del diagramma di ampiezza asintotico dopo la pulsazione $ omega_n$ ha pendenza $-40$ dB/dec.  
Il picco di risonanza si trova alla pulsazione (di risonanza) $ omega_r =  omega_n  sqrt(1-2 xi^2)$ con $|G_d (j omega_r)| =  dfrac(1,2|xi| sqrt(1-2 xi^2))$; alla frequenza $ omega_n$ si ha $|G_d (j omega_n)| =  dfrac(1,2|xi|)$
   
Soffermiamoci un attimo sul caso in cui $ xi  arrow 0$: se do una sinusoide con frequenza inferiore a $ omega_n$ essa non viene sfasata; se invece la sua frequenza è di poco superiore a $ omega_n$ la sua fase viene sfasata di $90^ degree$; il modulo viene amplificato di molto se la frequenza della sinusoide è nell'intorno di $ omega_n$.



=== Poli complessi coniugati a parte reale positiva
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

#cfigure("Images/Diagramma_poli_cc_pos.png", 68%)

Diagramma ottenuto da quello degli zeri (caso $ zeta<0$) ribaltando rispetto all'asse reale.



=== Ritardo temporale
Consideriamo $G(s) = e^(- tau s)$ con $G(j omega) = e^(-j  omega tau )$
$
    |G(j omega)|_("dB") &= 20  log |e^(-j  omega tau)|  
    \
    &= 20  log 1  
    \
    &=0
$
$
    arg(G(j omega)) &=  arg(e^(-j omega tau))  
    \
    &= -  omega tau
$
Questo tipo di sistema ritarda di $ tau$ il segnale in ingresso, quindi la fase viene attenuata mentre il modulo rimane invariato.

#cfigure("Images/Diagramma_rit_temp.png", 60%)



=== Proprietà bloccante degli zeri
Supponiamo di avere $G(s) =  mu dfrac(s^2+ alpha_n^2,(1+T_1 s)(1+T_2 s))$, con $T_1,T_2 > 0$ (quindi abbiamo un sistema asintoticamente stabile). Calcoliamo l'uscita del sistema supponendo un ingresso del tipo $u(t) = U  cos(omega_u t) $. La trasformata dell'ingresso è $U(s) =  dfrac(U s,s^2 +  omega_u^2)$


- Caso 1: $ omega_u  !=  alpha_n$.\  
    La trasformata dell'uscita sarà uguale a
    $
        Y(s) = G(s) U(s) =  mu  frac(U s (s^2+ alpha_n^2),(1+T_1 s)(1+T_2 s)(s^2 +  omega_u^2)) 
    $
    In base al denominatore, i modi presenti nell'uscita sono:
    #list(marker: [-],
        [$e^(-t slash T_1)$ dovuto al termine $1+T_1 s$],
        [$e^(-t slash T_2)$ dovuto al termine $1+T_2 s$],
        [$|G(j  omega_u)|U  cos( omega_u t +  arg(G(j omega_u)))$ dovuto al termine $s^2+ omega_u^2$]
    )
    
    
- Caso 2: $ omega_u =  alpha_n$.\  
    La trasformata dell'uscita sarà uguale a
    $
        Y(s) = G(s) U(s) =  mu  frac(U s  cancel((s^2+ alpha_n^2)),(1+T_1 s)(1+T_2 s) cancel((s^2 +  alpha_n^2))) =  mu  frac(U s,(1+T_1 s)(1+T_2 s))
    $
    In base al denominatore, i modi presenti nell'uscita sono:
    #list(marker: [-],
        [$e^(-t slash T_1)$ dovuto al termine $1+T_1 s$],
        [$e^(-t slash T_2)$ dovuto al termine $1+T_2 s$]
    )
    Pertanto nell'uscita $y(t) = k_1 e^(-t slash T_1) + k_2 e^(-t slash T_2)$ non sono presenti i modi corrispondenti agli zeri del sistema.



== Risonanza
Supponiamo di avere un sistema con poli immaginari coniugati $ plus.minus j  omega_n$, ovvero $G(s) =  mu  dfrac( omega_n^2,s^2+ omega_n^2)$ (rispetto al caso generale qui $ xi = 0$); il diagramma di Bode ha un picco di risonanza infinito alla pulsazione $ omega_n$.

Analizziamone il significato calcolando l'uscita del sistema in corrispondenza dell'ingresso $u(t) = U  cos( omega_u t)$. La trasformata dell'ingresso è $U(s) = U  dfrac(s,s^2+ omega_u^2)$, quindi quella dell'uscita è
$
    Y(s) = G(s)U(s) =  mu  frac(U  omega_n^2 s,(s^2 +  omega_n^2)(s^2 +  omega_u^2))
$

#set enum(numbering: "1)")




    + $ omega_u  !=  omega_n$
        $
            Y(s) =  frac(k_1,s-j  omega_n) +  frac( overline(k)_1,s+j  omega_n) +  frac(k_2,s-j  omega_u) +  frac( overline(k)_2,s+j  omega_u)
        $
        $
            y(t) = 2 |k_1|  cos( omega_n t +  arg(k_1)) + 2|k_2|  cos( omega_u t +  arg(k_2))
        $
        l'uscita è la somma di due sinusoidi a frequenza $ omega_n$ e $ omega_u$
    + $ omega_u =  omega_n$
        $
            Y(s) &= G(s) U(s)   
            \
            &=  mu U  frac(s  omega_n^2,(s^2+ omega_n^2)(s^2+ omega_n^2))   
            \
            &=  mu U  frac(s  omega_n^2,(s^2+ omega_n^2)^2)  
            \
            &=  frac(k_1,s-j omega_n) +  frac( overline(k)_1,s+j omega_n) +  frac(k_2,(s-j omega_n)^2) +  frac( overline(k)_2,(s+j omega_n)^2)
        $
        $
            y(t) = 2 |k_1|  cos( omega_n t +  arg(k_1)) + 2|k_2|  cos( omega_n t +  arg(k_2))
        $
        L'uscita tende a infinito per $t  arrow  infinity$, quindi il sistema non è BIBO stabile; questo fenomeno viene chiamato *risonanza*.

        #cfigure("Images/Risonanza.png", 35%)
        



== Azione filtrante dei sistemi dinamici
Quanto visto mostra come un sistema dinamico lineare e stazionario si comporta sostanzialmente come un _filtro_ per l'ingresso, modellandolo per produrre l'uscita.

=== Filtro passa-basso
Un filtro ideale passa-basso è un sistema che lascia passare inalterata, o amplificate di un valore costante, unicamente le armoniche del segnale di ingresso con pulsazione inferiore o uguale a un dato valore $ overline( omega)$; il diagramma di Bode del filtro è costante dino a $ overline( omega)$ e vale  $- infinity$ "dB" per $ omega >  overline( omega)$, mentre il diagramma della fase è nullo fino a $ overline( omega)$.  
La realizzazione di un filtro ideale passa-basso è di fatto impossibile, quindi definiamo un filtro reale passa-basso come un sistema con $G(j omega)$ che soddisfa le seguenti relazioni
$
    cases(
        dfrac(1, sqrt(2))  <=  dfrac(|G(j omega)|,|G(j 0)|)  <=  sqrt(2) wide& omega  <=  overline( omega)  
        \ \
        dfrac(|G(j omega)|,|G(j 0)|) <  dfrac(1, sqrt(2)) & omega>  overline( omega)
    )
$<passa_basso>

#cfigure("Images/Passa_basso.png", 70%)

N.B. Nella @eqt:passa_basso[] lo sfasamento introdotto da $G(s)$ non ha alcun ruolo, mentre il contributo di fase di un filtro #underline[reale] può essere significativo e non va trascurato.


=== Filtro passa-alto
Un _filtro ideale passa-alto_ è un sistema che lascia passare inalterate, o amplificate di una quantità costante, unicamente le armoniche del segnale di ingresso con pulsazione maggiore o uguale a $ tilde( omega)$.  

Un _filtro reale passa-alto_ è un sistema caratterizzato da una risposta in frequenza $G(j omega)$ il cui modulo soddisfa le seguenti relazioni
$
    cases(
        dfrac(|G(j omega)|,|G(j 0)|) <  dfrac(1, sqrt(2)) &  omega<  tilde( omega)  
        \ \
        dfrac(1, sqrt(2))  <=  dfrac(|G(j omega)|,|G(j 0)|)  <=  sqrt(2) wide& omega  >=  tilde( omega)
    )
$<passa_alto>

#cfigure("Images/Passa_alto.png", 70%)

Solo sistemi non strettamente propri possono avere le caratteristiche di un filtro passa-alto, dato che la @eqt:passa_alto[] implica che $G(j infinity)  != 0$.

=== Passa banda
I _filtri passa banda ideali_ sono sistemi che lasciano passare solo armoniche del segnali in ingresso comprese tra $ omega_(B 1)$ e $ omega_(B 2)$.

I _filtri passa banda reali_ sono sistemi che hanno una $G(j omega)$ che soddisfa le seguenti relazioni
$
    cases(
        dfrac(1, sqrt(2))  <=  dfrac(|G(j omega)|,|G(j omega_(max))|)  <=  sqrt(2) wide& omega  in [ omega_(B 1),  omega_(B 2)]  
        \ \
        dfrac(|G(j omega)|,|G(j omega_(max))|) <  dfrac(1, sqrt(2)) &  omega  in [0,  omega_(B 1)]  union [ omega_(B 2), 0]
    )
$<passa_banda>

#cfigure("Images/Passa_banda.png", 53%)



=== Elimina banda
I _filtri elimina banda ideali_ sono sistemi che non lasciano passare solo armoniche del segnali in ingresso comprese tra $ omega_(B 1)$ e $ omega_(B 2)$.  

I _filtri elimina banda reali_ sono sistemi che hanno una $G(j omega)$ che soddisfano le seguenti relazioni
$
    cases(
        dfrac(1, sqrt(2))  <=  dfrac(|G(j omega)|,|G(j omega_(max))|)  <=  sqrt(2)  wide&  omega  in [0,  omega_(B 1)]  union [ omega_(B 2), 0]  
        \ \
        dfrac(|G(j omega)|,|G(j omega_(max))|) <  dfrac(1, sqrt(2))  & omega  in [ omega_(B 1),  omega_(B 2)]
    )
$

#cfigure("Images/Elimina_banda.png", 53%)



== Esempio diagramma di Bode
Prendiamo una risposta in frequenza
$
    G(s) = 10 dot frac(1 + 0.1 s,1+10 s)
$
Il guadagno è $ mu = 10$, che in decibel è
$
    20  log  mu &= 20  log 10  \
    &= 20 "dB"
$
C'è uno zero 
$
    1+10^(-1)s & xarrow(width: #2em, "") s_z = -10 
    &space   
    omega_z &= 10
$ 
e un polo 
$
    1+10 s & xarrow(width: #2em, "") s_p = -10^(-1) 
    & space  
    omega_p &= 10^(-1)
$ 
$
    G(j omega) = 10 dot frac(1 + 10^(-1)j  omega,1 + 10 j  omega)
$
Il diagramma d'ampiezza risultante può essere visto come la somma dei contributi del polo e dello zero (in #text(green)[verde] il contributo dello zero, in #text(red)[rosso] il contributo dell polo, in #text(blue)[blu] la somma)

#cfigure("Images/Esempio_diagramma_amp.png", 60%)
#cfigure("Images/Esempio_diagramma_fase.png", 60%)

Per quanto riguarda il diagramma di fase, $-90^degree$ è un valore asintotico che si può raggiungere solo per $ omega  arrow  infinity$, quindi non sarà mai raggiunto in un sistema reale; inoltre la presenza di uno zero aumenta la fase nel diagramma.




=== Filtrare un'onda quadra con disturbo in alta frequenza
Consideriamo un segnale $s(t)$ ad onda quadra con periodo $T$ e ampiezza $A$:
$
    s(t) = A  "sign" (sin(frac(2  pi t,T) )  ) = 
    cases(
        A &  "se"  sin ( dfrac(2 pi t,T) )  >= 0   
        \
        -A wide&  "se"  sin ( dfrac(2 pi t,T) ) < 0
    )
$

#cfigure("Images/Filtro_onda_quadra.png", 42%)

Supponiamo che l'ingresso consista nell'onda quadra più un disturbo sinusoidale ad alta frequenza $ omega_N  >>  dfrac(2  pi,T)$ con ampiezza $A_N$
$
    u(t) = s(t) + A_N  sin( omega_N t)
$
Per filtrare il disturbo in alta frequenza possiamo usare un filtro passa-basso, cioè un sistema dinamico del primo ordine con un polo reale in $- dfrac(1,T_p)$: $G(s) =  dfrac(1,1+s T_p)$. Scegliendo opportunamente la costante di tempo $T_p$ il segnale in uscita sarà una versione filtrata dell'onda quadra con disturbo in alta frequenza quasi completamente attenuato.




= Sistemi di controllo: stabilità e prestazioni
== Schema di controllo in retroazione
Consideriamo il seguente schema di controllo in retroazione

#cfigure("Images/Retroazione_1.png", 60%)

Ci poniamo come obiettivo quello di garantire che l'uscita $y(t)$ segua il riferimento $w(t)$ (scelto dall'utente) in presenza di 

- disturbi (non misurabili) in uscita $d(t)$ e disturbi di misura $n(t)$
- incertezze sul modello $G(s)$ del sistema fisico (impianto) considerato
soddisfacendo opportune specifiche di prestazione.
   
Definiamo 
$
    L(s) = R(s) G(s)
$<funzione_anello>

come *funzione d'anello*, cioè la funzione di trasferimento in anello aperto del sistema.\  
$R(s)$ è chiamato *regolatore*.

=== Sistema in anello chiuso
Prendiamo un sistema in retroazione in anello chiuso (caso ideale senza rumore o disturbi)

#cfigure("Images/Retroazione_anello_chiuso.png", 65%)

*N.B.* Chiameremo sempre la $w(t)$ come uscita di riferimento $y_("RIF")(t)$; quindi $y_("RIF")(t)$ è quello che voglio ottenere dal sistema, mentre $y(t)$ è come si comporta il sistema.
$
    Y(s) = F(s)  underbrace(Y_("RIF")(s),W(s))
$<relazione_ingresso_riferimento-uscita>
è ovvio che se potessimo decidere il valore di $F(s)$ sarebbe 1.


== Schema generale e semplificazione
Lo schema del sistema in retroazione ad anello aperto cattura anche strutture più complesse che includono attuatori e trasduttori

#cfigure("Images/Schema_generale_retro.png", 60%)

*N.B.* Il riferimento $w$ viene filtrato con una replica della dinamica del sensore $T(s)$ in modo che sia "compatibile" con la dinamica dell'uscita $y$ retroazionata.
   
Usando le proprietà di schemi a blocchi interconnessi, si può riscrivere lo schema precedente in modo equivalente.

#cfigure("Images/Schema_gen_rielaborato.png", 60%)

Così lo schema diventa

#cfigure("Images/Schema_gen_semp.png", 60%)

- Sistemi: $R(s) = T(s)  tilde(R)(s),   G(s) = A(s)  tilde(G)(s)$
- Segnali: $W(s) =  tilde(W)(s),   N(s) = T^(-1)(s) tilde(N)(s),   D(s) = D_a (s) tilde(G)(s) + D_u(s)$

il disturbo sull'attuatore $d_a (t)$ viene filtrato dal sistema. Bisogna tenerne conto quando si fanno considerazioni sul disturbo in uscita $d(t)$.  

$R(s)$ è la funzione di trasferimento del regolatore e $G(s)$ del sistema sotto controllo; inoltre assumiamo che $R(s)$ e $G(s)$ siano funzioni razionali, con $G(s)$ strettamente propria, mentre $R(s)$ può essere rappresentativa di sistemi non strettamente propri e non dinamici. La funzione ad anello $L(s) = R(s)G(s)$ risulta sempre strettamente propria.


== Disaccoppiamento frequenziale dei segnali
Nelle applicazioni di interesse ingegneristico tipicamente le bande dei segnali di ingresso $w(t),   d(t),   n(t)$ sono limitate in opportuni range.

#cfigure("Images/Disacc_freq_segnali.png", 60%)

- $w(t),   d(t)$ hanno bande a "basse frequenze", ad esempio  posizioni, rotazioni, velocità, etc ... di sistemi meccanici
- $n(t)$ ha bande ad "alte frequenze", ad esempio disturbi termici in componenti elettronici, accoppiamenti con campi elettromagnetici, etc ... 

Lo stesso vale per le trasformate: $W(j omega), D(j omega)$ hanno valori non nulli a "basse frequenze", mentre $N(j omega)$ ha valori non nulli ad "alte frequenze".



== Requisiti di un sistema di controllo
=== Stabilità
*Stabilità in condizioni nominali* Il requisito fondamentale per il sistema in retroazione che stiamo considerando è l'asintotica stabilità (@Stabilità_interna[]) o stabilità BIBO (@BIBO[]).

*Stabilità robusta* La stabilità deve essere garantita anche in condizioni perturbate (errori di modello o incertezze nei parametri), perché la $G(s)$ rappresenta soltanto un modello approssimato del sistema sotto controllo.    


=== Prestazioni
Oltre alla stabilità, è necessario che il sistema abbia determinate prestazioni, cioè che abbia le proprietà elencate.
   
*Prestazioni statiche in condizioni nominali* Il sistema ha un errore $e$ limitato o nullo per $t  arrow  infinity$, cioè dopo che si è esaurito il transitorio iniziale, a fronte di ingressi $w,   d,   n$ con determinate caratteristiche.

Esempi:

- errore in risposta a un ingresso a gradino (transizione ad un nuovo riferimento o disturbi costanti su attuatori/sensori) o rampa;
- risposta a un ingresso sinusoidale a date frequenze (disturbi con certe componenti frequenziali)

*Prestazioni dinamiche in condizioni nominali* Prestazioni del sistema in transitorio relative a 

- risposta a un riferimento $w$, data in termini di tempo di assestamento $T_(a, epsilon)$ e sovraelongazione $S %$ massimi;
- risposta a disturbi $d$ e $n$, data in termini di attenuazione in certi range di frequenze (bande di frequenza dei disturbi);
- moderazione della variabile di controllo $u$, data in termini di contenimento dell'ampiezza (per evitare saturazione di attuatori, uscita da range in cui la linearizzazione è valida o costi eccessivi).




=== Stabilità robusta del sistema retroazionato (Criterio di Bode)
Poiché la stabilità di un sistema lineare non dipende dagli ingressi, consideriamo il seguente schema a blocchi

#cfigure("Images/Stabilita_robusta.png", 60%)

Per studiare la stabilità robusta, in presenza di incertezze, del sistema retroazionato ci baseremo su un principio fondamentale: il *Criterio di Bode*, che lega la stabilità del sistema retroazionato a quella del sistema in anello aperto.
$
    F(s) &=  frac(Y(s),W(s))  ==> Y(s) = F(s) W(s)
$
Tenendo conto che, logicamente, l'errore del sistema $e(t)$ è la differenza tra il l'uscita di riferimento $y_("RIF") (t)$ e l'uscita reale $y(t)$
$
    Y(s) &= R(s) G(s)  overbrace(E(s), cal(L)[e(t)])  
    \
    &=R(s)G(s)  (W(s) - Y(s) )  
    \
    &= R(s)G(s) W(s) - R(s) G(s)Y(s)  
$
Riarrangiamo i termini
$
    Y(s) + R(s)G(s)Y(s) &= R(s)G(s)W(s)  
    \
    (1+R(s)G(s) )Y(s) &= R(s)G(s)W(s)
$
quindi
$
    Y(s) =  underbrace( frac(R(s)G(s),1+R(s)G(s)),F(s)) W(s)
$
$
    cases(reverse: #true,
        F(s) =  dfrac(R(s)G(s),1+R(s)G(s))  
        \ 
        L(s) = R(s)G(s)
    )
    ==>
    F(s) =  frac(L(s),1+L(s))
$



== Margini di fase e ampiezza
=== Margine di fase
In un sistema ad anello chiuso la funzione di trasferimento è $dfrac(L(s), 1+L(s))$. Sappiamo che un sistema è instabile se la funzione di trasferimento va a $infinity$; quindi nel nostro caso
$
    frac(L(s), 1+L(s)) = infinity <==> 1+L(s) = 0 <==> L(s) = -1
$
La funzione $L(s) = -1$ rappresenta un guadagno del sistema pari a 1, o 0 dB, e una fase di $-180 degree$ (riproduce l'input ribaltato). Per questo se nel diagramma di Bode di un sistema, in corrispondenza dello 0 dB del diagramma delle ampiezze, il diagramma di fase vale $-180 degree$ il sistema è instabile.\
Il margine di fase quindi, esprime quanto siamo lontani dall'instabilità, ed è definito come

$
    M_f = 180^( degree) +  arg(L(j omega_c)) "con"  omega_c "tale che" |L(j omega_c)|_("dB") = 0 
$<margine_fase>


Nota: $M_f =  arg(L(j omega_c)) - (-180^( degree)) = 180^( degree) +  arg(L(j omega_c))$.  

La seguente figura (e questa risorsa: #link("https://youtu.be/ThoA4amCAX4")[Gain and Phase Margins Explained]) può aiutare a comprendere meglio cos'è il margine di fase (in figura il margine di fase è indicato con $ phi_m$)

#cfigure("Images/Margine_fase.png", 65%)

Il margine di fase indica il ritardo massimo che può essere introdotto nel sistema mantenendolo stabile.
  
Consideriamo un sistema che ritarda il suo input di $ tau$, che quindi ha funzione di trasferimento $e^(-s tau)$. Il suo diagramma di Bode delle ampiezze è costante a 0 dB. Lo sfasamento è $- omega tau$, che nel diagramma di Bode delle fasi, in scala semi-logaritmica, ha un andamento di tipo esponenziale.  
Se $L(s) = e^(-s  tau)  tilde(L)(s)$ la pulsazione critica $ omega_c$ non cambia, è la stessa per entrambe.  
Un ritardo quindi riduce il margine di fase in quanto, per $ omega =  omega_c$, riduce la fase:
$
     arg(L(j omega_c)) =  arg  ( tilde(L)(j omega_c) ) -  tau  omega_c
$
quindi per essere asintoticamente stabile, un sistema deve poter tollerare un ritardo $ tau$ che soddisfi la disequazione
$
     tau <  frac(M_f, omega_c)
$

#cfigure("Images/Margine_fase_2.png", 90%)



=== Margine di ampiezza
La definizione di margine di ampiezza parte dallo stesso assunto del margine di fase, solo che in questo caso prendiamo come riferimento la frequenza alla quale il diagramma delle fasi ha valore $-180 degree$. Infatti anch'esso ci da una misura di quanto siamo distanti dall'instabilità.

#align(center)[$M_a = -|L(j omega_( pi))|_("dB")$ con $ omega_( pi)$ tale che $ arg(L(j omega_( pi))) = -180^( degree)$]

La seguente figura può aiutare a comprendere meglio cos'è il margine di ampiezza (in figura il margine di ampiezza è indicato con $|k_m|_("dB")$)

#cfigure("Images/Margine_ampiezza.png", 60%)

Il margine di ampiezza indica il guadagno massimo che può essere introdotto mantenendo il sistema asintoticamente stabile.

Supponendo di introdurre un ulteriore guadagno $k$ nel sistema, esso rimane asintoticamente stabile per tutti i valori di $k$ inferiori a $M_a$.

#cfigure("Images/Margine_ampiezza_2.png", 90%)




=== Casi patologici
Ci sono casi in cui $M_f$ e $M_a$ non sono definiti o non sono informativi:
- nel caso di *intersezioni multiple*, in cui il diagramma delle ampiezze attraversa l'asse a 0 dB più di una volta;
- nel caso di *assenza di intersezioni*, in cui il diagramma delle ampiezze non attraversa mai l'asse a 0 dB;
- nel caso di margini di ampiezza e fase con *segni discordi*, perché per essere informativi $M_f$ e $M_a$ devono avere lo stesso segno.



== Criterio di Bode
Si supponga che
+ $L(s)$ non abbia poli a parte reale (strettamente) positiva
+ il diagramma di Bode del modulo di $L(j omega)$ attraversi una sola volta l'asse a 0 dB.

Allora, condizione necessaria e sufficiente perché il sistema retroazionato sia asintoticamente stabile è che risulti $ mu > 0$ (con $ mu$ guadagno statico di $L(j omega)$) e $M_f > 0$.
   
In questo modo quindi la stabilità del sistema in retroazione è determinata dalla lettura di un solo punto sul diagramma di Bode di $L(j omega)$. Si rammenta che $M_f$ e $M_a$ in genere vanno considerati simultaneamente e forniscono una misura della robustezza rispetto a incertezze su $L(s)$.


#pagebreak()
== Funzioni di sensitività

#cfigure("Images/Schema_gen_semp.png", 65%)

Ingressi del sistema in anello chiuso:
- $w(t)$ riferimento (andamento desiderato per $y(t)$)
- $d(t)$ disturbo in uscita
- $n(t)$ disturbo di misura

Uscite di interesse:
- $e(t) = w(t) - y(t)$ errore di inseguimento
- $y(t)$ uscita controllata
- $u(t)$ ingresso di controllo del sistema in anello aperto (impianto)

Definiamo le _funzioni di sensitività_ come funzioni di trasferimento tra ingressi e uscite di interesse.
$
    cases(
        S(s) =  dfrac(1,1+R(s)G(s)) & "Funzione di sensitività"  
        \ \
        F(s) =  dfrac(R(s)G(s),1+R(s)G(s)) & "Funzione di sensitività complementare"  
        \ \
        Q(s) =  dfrac(R(s),1+R(s)G(s)) wide& "Funzione di sensitività del controllo"
    )
$<funzioni_sensitivita>

$
    mat(delim: "[",
        Y(s)  ;  U(s) ;  E(s)
    )
    =
    mat(delim: "[",
        F(s) , S(s) , -F(s);  
        Q(s) , -Q(s) , -Q(s) ; 
        S(s) , -S(s) , F(s)
    )
    mat(delim: "[",
        W(s) ; D(s)  ;  N(s)
    )
$

#cfigure("Images/Funzioni_sens.png", 60%)

Definiamo $y_w (t)$ l'uscita con ingresso $w(t)$, $y_d (t)$ l'uscita con ingresso $d(t)$ e $y_n (t)$ l'uscita con ingresso $n(t)$; per il principio di sovrapposizione degli effetti
$
    y(t) = y_w (t) + y_d (t) + y_n (t)
$


=== Funzione di sensitività complementare
Prendiamo in considerazione solo l'ingresso $w(t)$. 

Sappiamo dalla @eqt:relazione_ingresso_riferimento-uscita[] che la funzione di trasferimento che lega $Y_w (s)$ e $W(s)$ è $F(s)$
$
    Y_w (s) = F(s) W(s)
$
e sappiamo che la $F(s)$ è così definita
$
    F(s) =  frac(R(s)G(s),1+R(s)G(s)) :=  "Funzione di sensitività complementare"
$<funzione_sensitivita_complementare>

=== Funzione di sensitività
Prendiamo in considerazione solo l'ingresso $d(t)$.

#cfigure("Images/Funzione_di_sensitività.png", 65%)

$
    Y_d (s) &=  underbrace(D(s), cal(L)[d(t)]) + R(s) G(s)  underbrace(E_d (s), cal(L)[e(t)])  
    \
    &= D(s) + R(s) G(s)  (0 - Y_d (s) ) wide& #text(green)[$ <== E_d (s) =  underbrace(Y_w (s),0) - Y_d (s) $]
$
riarrangiamo i termini
$
    Y_d (s) + R(s) G(s) Y_d (s) &= D(s) 
    \ 
    (1+R(s) G(s) ) Y_d (s) &= D(s)
$
quindi
$
    Y_d (s) =  frac(1,1+R(s) G(s))D(s)
$
$
    S(s) =  frac(1,1+R(s) G(s)) := "Funzione di sensitività"
$<funzione_sensitivita>

siccome $L(s) = R(s) G(s)$ la funzione di sensitività può essere scritta come
$
    S(s) =  frac(1,1+L(s))
$<funzione_sensitivita_2>


=== Funzione di sensitività del controllo
Prendiamo in considerazione solo l'ingresso $n(t)$.

#cfigure("Images/Funzione_di_sensitività_controllo.png", 65%)

$
    Y_n (s) &= F(s)  (-N(s) )  
    \
    &=-F(s)N(s)
$




=== Considerazioni
*Stabilità* Il denominatore di tutte le funzioni di sensitività è lo stesso. Si ricordi che la stabilità è determinata dai poli della funzione di trasferimento.  Questo è consistente con il fatto che la stabilità del sistema (retroazionato) non dipende dal particolare ingresso considerato.

$
    Y(s) &= Y_w (s) + Y_d (s) + Y_n (s)  
    \
    &= F(s)W(s) + S(s)D(s) - F(s)N(s)
$
Per seguire fedelmente il riferimento $w(t)$ vorremmo $F(s) = 1$, che è il caso ideale
$
    Y_n (s) &= -N(s) 
    &space space  
    S(s) &=  frac(1,1+L(s))
$
e per annullare l'effetto del disturbo $d(t)$ vorremmo $S(s) = 0$. Tuttavia il disturbo $n(t)$ non sarebbe per niente attenuato.
   
Inoltre $S(s)+F(s) = 1$ sempre
$
    S(s)+F(s) &=  frac(1,1+L(s)) +  frac(L(s),1+L(s))  
    \
    &= 1 
$
è il motivo per cui $F(s)$ viene chiamata funzione di sensitività complementare.
   
Noi lavoreremo in  frequenze in modo da avere, per $F(j omega) =  dfrac(L(j omega),1+L(j omega))$
- $|F(j omega)|  approx 1$ a "basse" frequenze (inseguimento di $w(t)$)
- $|F(j omega)|  approx 0$ ad "alte" frequenze (abbattimento di $n(t)$)

Quindi progettermo $R(j omega)$ in mode che
- $|L(j omega)|  >> 1$ a basse frequenze
- $|L(j omega)|  << 1$ ad alte frequenze

Ricordando che
- $w(t), d(t)$ hanno componenti frequenziali a "basse" frequenze
- $n(t)$ ha componenti frequenziali ad "alte" frequenze


=== Errori
$
    E_w (s) &= W(s) - Y_w (s)  
    & 
    E_d (s) &= W(s) - Y_d (s)
    space&
    E_n (s) &= W(s) - Y_n (s) 
    \
    &= W(s) - F(s)W(s)  
    &
    &= 0 - Y_d (s) 
    &
    &= 0 - Y_n (s) 
    \
    &=  (1-F(s) ) W(s)  
    &
    &= -Y_d (s) 
    &
    &= -Y_n (s)  
    \
    #text(fill: green, size: 8pt, baseline: -1pt)[$S(s) = 1 - F(s) ==>$]&= S(s) W(s) &  space
    &= -S(s)D(s)
    &
    &= -  (-F(s)N(s) )
    \
    &&&&
    &=F(s) N(s)
$



=== Analisi in frequenza della funzione di sensitività complementare <analisi_funzione_sensitivita_complementare>
$
    F(s) =  frac(L(s),1+L(s))
$
passiamo alle frequenze
$
    |F(j omega)| &=  lr(|  frac(L(j omega),1+L(j omega)) |)
    \   
    &=  frac(|L(j omega)|,|1+L(j omega)|)  
    \
    ==>_("scelta di" \ "design su " R(j omega))
    & approx 
    cases(
        1 & omega  <<  omega_c \ 
        |L(j omega)| wide& omega  >>  omega_c
    )
$
Consideriamo $ omega_c$ pulsazione di taglio
$
    |F(j omega)|_("dB")  approx
    cases(
        0  "dB" & omega  <=  omega_c  \
        |L(j omega)|_("dB") wide& omega >  omega_c
    )
$

#cfigure("Images/Analisi_sens_compl.png", 65%)



=== Analisi in frequenza della funzione di sensitività <analisi_funzione_sensitivita>
$
    S(s) &=  frac(1,1+L(s))
    space space &
    |S(j omega)| &=  frac(1,|1+L(j omega)|)
    \
    &&
    & approx
    cases(
        dfrac(1,|L(j omega)|) wide& omega  <=  omega_c  
        \ \
        1 &  omega >  omega_c
    )
    space&
    
$
#v(2em)
$
    |S(j omega)|_( "dB")  approx 
    cases(
        - |L(j omega)|_( "dB") wide& omega  <=  omega_c 
        \ 
        0  "dB" & omega >  omega_c
    )
$

#cfigure("Images/Analisi_sens.png", 60%)




=== Analisi in frequenza della funzione di sensitività del controllo <analisi_funzione_sensitivita_controllo>
$
    Q(s) &=  frac(R(s),1+R(s) G(s))
    space space&
    |Q(j omega)| &=  lr(|  frac(R(s),1+R(s) G(s))  |)
    \
    &&
    & approx
    cases(
        dfrac(1,G(j omega)) wide& omega <=  omega_c  
        \ \
        R(j omega) & omega >  omega_c
    )
$
#v(2em)
$
    |Q(j omega)|_("dB")  approx 
    cases(
        -|G(j omega)|_("dB")  wide& omega <=  omega_c  
        \ \
        |R(j omega)|_( "dB") & omega >  omega_c
    )
$
#cfigure("Images/Analisi_sens_controllo.png", 60%)

A basse frequenze il modulo di $Q(j omega)$ dipende da $G(j omega)$, quindi non possiamo influenzarlo con il regolatore. Occorre evitare valori di $ omega_c$ troppo elevati.



=== Poli complessi coniugati di $bold(F(s))$ (sensitività complementare) <poli_complessi_coniugati_sens_complementare>
La funzione di sensitività complementare può avere una coppia di poli c.c. dominanti
$
    L(s) =  frac(20,(1+10s)(1+2s)(1+0.2s))
$

#cfigure("Images/Poli_cc_F(s).png", 65%)

Mettiamo in relazione il picco di risonanza di $F (j omega)$ con lo smorzamento $ xi$ associato, assumendo che $ omega_n  approx  omega_C$.
$
    |F(j omega)|  approx 
    cases(
        0  "dB" & omega  <=  omega_c  
        \
        |L(j omega)|_("dB") wide& omega >  omega_c
    )
$
Approssimazione di $F(s)$ a poli complessi coniugati dominanti:

#cfigure("Images/Poli_sensitivita_complementare.png", 65%)

Assumendo $omega_n approx omega_c$, la funzione di sensitività complementare vale (si veda la @poli_complessi_coniugati_parte_reale_negativa[])
$
    |F(j omega_(c))| &approx 1/(2 xi)
$
dove $xi$ è lo smorzamento dei poli complessi coniugati.

Altrimenti, tenendo conto che il modulo di $|L(j omega)|$ alla frequenza di taglio $omega_c$, come si vede dal grafico, vale $0$ dB (quindi 1)
$
    |F(j omega_c)| &= frac(overbrace(|L(j omega_c)|, 1), |1 + L(j omega_c)|)
    \
    &= frac(1, |1 + underbrace(L(j omega_c), #text(10pt)[$1 dot e^(j phi_c)$])|)
    \
    &= frac(1, |1+ cos phi_c + j sin phi_c|)
    \
    &= frac(1,sqrt((1+ cos phi_c)^2 + (sin phi_c)^2))
    \
    &= frac(1, sqrt(1+ cos^2 phi_c + sin^2 phi_c + 2 cos phi_c ))
    \
    &= frac(1, sqrt(2(1+cos phi_c)))
$
dalla trigonometria sappiamo che $cos(phi_c) = - cos(pi + phi_c)$ e, ricordando la definizione di @eqt:margine_fase[margine di fase]
$
    frac(1, sqrt(2(1+cos(phi_c)))) 
    &= frac(1, sqrt(2(1-cos M_(f)^("rad")) ) )
    \
    #text(fill: green, baseline: -1pt, size: 9pt)[$1 - cos alpha = 2 sin^2 frac(alpha,2) ==>$]
    &= frac(1, sqrt(4 sin^2dfrac(M_(f)^("rad"), 2) ) )
    \
    &= frac(1,2 sin dfrac(M_(f)^("rad"), 2))
$

Uguagliando le due espressioni si ha
$
    xi = sin frac(M_(f)^("rad"), 2) approx frac(M_(f)^("rad"), 2)
$
si è potuto approssimare il seno perché la pulsazione di taglio è molto bassa.

Convertendo i radianti in gradi
$
    xi &= frac(M_f, 2) frac(pi, 180)
    \
    &= M_f dot frac( pi, 360)
    \
    &= M_f dot frac(3.14, 3.6 times 100)
    \
    & approx frac(M_f, 100)
$
questa equazione mette in relazione la $F(j omega)$ e la $L(j omega)$.



== Analisi statica: errore a un gradino
Sia $e_infinity = display(lim_(t -> infinity)) e(t)$ con $e(t) = w(t) - y(t)$ errore in risposta a un gradino $w(t) = W 1(t)$.

Utilizzando il @teorema_valore_finale[teorema del valore finale]

$
    e_(infinity) &= lim_(s -> 0) s E(s) \
    #text(fill: green, size: 9pt)[$E_w (s) = S(s) W(s) ==>$] &= lim_(s -> 0) s S(s)W(s)\
    #text(fill: green, size: 9pt)[$cal(L)[W 1(t)] = W/s ==>$] &= lim_(s -> 0) s S(s)W/s\
    &= W lim_(s -> 0)S(s)
$

Sia $L(s) = dfrac(N_L (s), D_L (s)) = dfrac(N_L (s), s^g D'_L (s))$ con $N_L (0) = mu$ e $D'_L (0) = 1$, allora

$
    lim_(s -> 0)S(s) &= lim_(s->0) frac(D_L (s), N_L (s) + D_L (s)) \
    &= lim_(s->0) frac(s^g D'_L (s), N_L (s) + s^g D'_L (s)) \
    &= frac(s^g, mu + s^g)
$

Si ha quindi
$
    e_infinity = W lim_(s->0) frac(s^g, mu + s^g) =
    cases(
        dfrac(W, 1+mu) wide& g=0 \
        0 &g>0
    )
$

== Analisi statica: errore a ingressi $frac(W, s^k)$

Sia $e_infinity = display(lim_(t -> infinity)) e(t)$ con $e(t) = w(t) - y(t)$ errore in risposta a un ingresso con trasformata $W(s) = dfrac(W, s^k)$
$
    e_infinity &= lim_(s -> 0) s S(s) frac(W, s^k) \
    &= W lim_(s->0) frac(s^(g-k+1), mu + s^g) = 
    cases(
        infinity wide& g<k -1 \
        dfrac(W,mu) &g=k-1 \
        0 & g>k-1
    )
$
Quindi 
- se $g< k-1$ l'errore diverge
- se $g=k-1$ l'errore a regime è finito e diminuisce all'aumentare di $mu$
- se $g>k-1$ l'errore a regime è nullo

*Nota:* le precedenti valutazioni sono valide solo se il sistema in anello chiuso è asintoticamente stabile.

Affinché l'errore a regime a $W(s) = dfrac(W, s^k)$ sia nullo occorre che $L(s)$ abbia un numero di poli almeno pari a $k$ (principio del modello interno).

== Principio del modello interno
Possiamo generalizzare il risultato precedente come segue.

Affinché un segnale di riferimento (rispetto un disturbo di misura) con una componente spettrale alla frequenza $omega_0$ sia inseguito a regime perfettamente in uscita è necessario e sufficiente che
- il sistema chiuso in retroazione sia asintoticamente stabile;
- il guadagno d'anello $L(s)$ abbia una coppia di poli c.c. sull'asse immaginario con pulsazione naturale pari a $omega_0$.




= Sistemi di controllo: progetto del regolatore
Consideriamo il seguente schema di controllo in retroazione:

#cfigure("Images/Regolatore_1.png", 67%)

=== Riepilogo specifiche

*Stabilità robusta rispetto a incertezze*\
 Stabilità in presenza di errori di modello o incertezze di parametri.

*Precisione statica*\
 Sia $e_infinity = display(lim_(t -> infinity)) e(t)$ il valore a regime dell'errore in risposta a riferimenti $w(t)$ o disturbi in uscita $d(t)$ "canonici"; la specifica da seguire è
 $
 |e_(infinity)| <= e^star &space "oppure" & space e_infinity = 0
 $

*Precisione dinamica*\
 Tipicamente specifiche in termini di sovraelongazione e tempo di assestamento massimi; le specifiche da seguire sono
 $
 S% <= S^star &space T_(a, epsilon) <= T^star
 $

*Attenuazione disturbo in uscita*\
 Il disturbo in uscita $d(t)$, con una banda limitata in un range di pulsazioni $[w_(d,min), w_(d,max)]$, deve essere attenuato di $A_d$ dB ($A_d > 0$).2

*Attenuazione disturbo di misura*\
 Il disturbo di misura $n(t)$, con una banda limitata in un range di pulsazioni $[w_(n,min), w_(n,max)]$, deve essere attenuato di $A_n$ dB ($A_n > 0$).

*Nota:* in applicazioni ingegneristiche in genere $w_(d,max) << omega_(n,min)$

*Moderazione variabile di controllo $u(t)$*\
 Contenimento dell'ampiezza della variabile di controllo $u$ in ingresso al sistema fisico (impianto).

*Fisica realizzabilità del regolatore $R(s)$*\
 Il regolatore deve essere un sistema proprio, quindi il grado relativo (differenza tra poli e zeri) deve essere maggiore o uguale a zero.



== Specifiche in termini di guadagno d'anello

=== Stabilità robusta rispetto a incertezze
 Stabilità in presenza di errori di modello o incertezze di parametri; ad esempio massimo ritardo temporale $tau_("max")$ o massima incertezza sul guadagno statico $Delta mu_"max"$.

=== Specifica su $L(j omega)$
 $
    M_f >= M_f^star
 $

=== Precisione statica
 Per soddisfare tali specifiche va considerata l'analisi statica effettuata sulla funzione di sensitività $S(s)$.\
 Ad esempio: $|e_infinity| <= e^star$ in risposta a un gradino $w(t) = W 1(t), thick d(t) = D 1(t)$ con $|W|<= W^star$ e $|D|<= D^star$.
 $
 e_infinity &= frac(W,1+mu) + frac(D,1+mu)\
            &=frac(D+W,1+mu)\
            &approx frac(D+W,mu)

 $

 $
 mu = L(0) >= frac(D^star+W^star,e^star)
 $
#v(5pt)
Altro esempio: $e_(infinity) = 0$ in risposta a $W(s) = dfrac(W,s^k)$ e/o $D(s) = dfrac(D,s^k)$
#v(3pt)
#align(center)[$L(s)$ deve avere $k$ poli nell'origine]
#v(5pt)
Se $|e_infinity| <= e^star$ in riposta a $W(s) = dfrac(W,s^k)$ e $D(s) = dfrac(D,s^k)$ allora
#v(3pt)
#align(center)[$k-1$ poli in $L(s)$ e $mu >= dfrac(D^star+W^star, e^star)$]
#v(3pt)
Se $e_infinity = 0$ in risposta a un disturbo sull'attuatore $D_a (s) = dfrac(D_a,s^k)$, allora 
$
    D(s) = D_a (s) G(s) &space space E(s) = S(s)G(s)D_a (s)
$
quindi
#align(center)[$k$ poli nell'origine in $R(s)$]


=== Precisione dinamica
Specifiche: $S% <= S^star$ e $T_(a,epsilon) <= T^star$.

Se progettiamo $L(j omega)$ in modo che $F(j omega)$ abbia una coppia di poli complessi coniugati dominanti in $omega_n approx omega_c$ con coefficiente di smorzamento $xi$ allora, come abbiamo visto nella @poli_complessi_coniugati_sens_complementare[]
$
    xi approx frac(M_f,100)
$

Perché $S% <= S^star$ allora $xi >=xi^star$, con $S^star = e^(frac(-pi xi^star,sqrt(1-(xi^star)^2)))$, e quindi
$
    M_f >= 100 xi^star
$

Perché $T_(a,1) <= T^star$ allora, ricordando la @eqt:tempo_assestamento[], e sapendo che $T = dfrac(1, xi omega_n)$, $xi omega_n >= dfrac(4.6, T^star)$
$
    M_f omega_c >= frac(460, T^star)
$

#cfigure("Images/Specifiche_dinamiche.png", 70%)
La zona proibita per il diagramma di fase va evitata *solo a $omega_c$*.


=== Attenuazione disturbo in uscita $d(t)$
Il disturbo in uscita $d(t)$, con una banda limitata in un range di pulsazioni $[omega_(d,min) , omega_(d,max)]$, deve essere attenuato di $A_d$ dB. (Nota: $A_d$ > 0).

Ricordiamo che se $d(t) = D cos(omega t + phi)$ allora
$
    y(t) = |S(j omega)|D cos(omega t + phi + arg(S(j omega)))
$
e che, grazie all' @analisi_funzione_sensitivita[analisi in frequenza della funzione di sensitività]
$
    |S(j omega)|_"dB" approx
    cases(
        -|L(j omega)|_"dB" &wide omega <= omega_c\
        0 &wide omega > omega_c
    )
$
Da specifica vogliamo $|S(j omega)|_"dB" ≤ -A_d "dB"$. Poiché $omega_(d,max) ≪ omega_c$ si ha
$
    |L(j omega)|_"dB" >= A_d "dB"
$
Ad esempio, se $d(t)$ deve essere attenuato di 20 dB allora $|L(j omega)|_"dB">= 20 "dB"$.

#cfigure("Images/Attenuazione_disturbo_uscita.png", 54%)


=== Attenuazione disturbo di misura $n(t)$
Il disturbo di misura $n(t)$, con una banda limitata in un range di pulsazioni $[omega_n,min , omega_n,max]$, deve essere attenuato di $A_n$ dB.

Ricordiamo che se $n(t) = N cos(omega t + phi)$ allora
$
    y(t) = |F (j omega)|N cos(omega t + φ - arg(F (j omega)))
$
e che, grazie all' @analisi_funzione_sensitivita_complementare[analisi in frequenza della funzione di sensitività complementare]
$
    |F (j omega)|_"dB" approx
    cases(
        0 &wide omega <= omega_c\
        |L(j omega)|_"dB" &wide omega > omega_c
    )
$
Da specifica vogliamo $|F (j omega)|_"dB" <= -A_n "dB"$. Poiché $omega_(n,min)>> omega_c$, si ha
$
    |L(j omega)|_"dB" <= -A_n "dB"
$
Ad esempio se $n(t)$ deve essere attenuato di 20 dB allora $|L(j omega)|_"dB" <= -20 "dB"$.
#cfigure("Images/Attenuazione_disturbo_misura.png", 50%)


=== Moderazione variabile di controllo $u(t)$
Contenimento dell'ampiezza della variabile di controllo $u$ in ingresso al sistema fisico (impianto).

Ricordiamo che se $w(t) = W cos(omega t + phi)$ allora
$
    u(t) = |Q(j omega)|W cos(omega t + phi + arg(Q(j omega)))
$
e che, grazie all' @analisi_funzione_sensitivita_controllo[analisi in frequenza della funzione di sensitività del controllo]
$
    |Q(j omega)|_"dB" approx
    cases(
        -|G(j omega)|_"dB" &wide omega <= omega_c\
        |R(j omega)_"dB" &wide omega > omega_c
    )
$
Poiché vogliamo contenere $|Q(j omega)|_"dB"$ e non abbiamo controllo su $G(j omega)$ dobbiamo
- limitare $omega_c$;
- realizzare $R(j omega)$ passa-basso.
#cfigure("Images/Moderazione_u.png", 70%)
/*-------------RIVEDI-----------------*/
Il limite superiore su $omega_c$ può essere determinato dalle specifiche sulla variabile di controllo $u(t)$.


=== Fisica realizzabilità del regolatore
Il regolatore deve essere un sistema proprio, quindi il grado relativo (differenza poli-zeri) deve essere maggiore o uguale a zero.

A pulsazioni elevate la pendenza $-k_L$ dB/dec di $|L(j omega)|_"dB"$ è determinata dalla differenza tra poli (ciascuno contribuisce con pendenza -20dB/dec) e zeri (ciascuno contribuisce con pendenza 20dB/dec).

Se a pulsazioni elevate $|G(j omega)|_"dB"$ ha pendenza $-k_G$ dB/dec allora
$
    -k_L <= -k_G
$
perché sappiamo che $L(s) = R(s) G(s)$ e che i poli (che fanno diminuire la pendenza) di $R(s)$ sono maggiori o uguali agli zeri, quindi moltiplicando $R(s)$ con $G(s)$ otteniamo una funzione di trasferimento con una pendenza uguale o minore di quella di $G(s)$.



=== Riepilogo specifiche
#cfigure("Images/Riepilogo_specifiche.png", 80%)



== Sintesi del regolatore
=== Loop Shaping
Il _loop shaping_, o sintesi per tentativi, consiste nel "dare forma" alla $L(j omega)$ in modo che:
- il diagramma delle ampiezze non attraversi le regioni proibite in bassa e alta frequenza;
-  per $omega = omega_c$ rispetti il vincolo sul margine di fase.
procedendo per tentativi basati su opportune considerazioni.


== Struttura del regolatore
È conveniente dividere il progetto in due fasi fattorizzando $R(s)$ come
$
    R(s) = R_s (s) R_d (s)
$
cioè come prodotto di due regolatori, uno statico e uno dinamico.

*Regolatore statico:*
$
    R_s (s) = mu_s/s^k
$
progettato per soddisfare precisione statica e attenuazione dei disturbi $d(t)$.


*Regolatore dinamico:*
$
    R_d (s) = mu_d frac(product_i (1+tau_i s) product_i (1+2 frac(zeta_i, alpha_(n,i))s + frac(s^2,alpha^2_(n,i))), product_i (1+T_i s) product_i (1+2 frac(xi_i, omega_(n,i))s + frac(s^2,omega^2_(n,i))))
$
progettato per soddisfare stabilità robusta, precisione dinamica, attenuazione dei disturbi $n(t)$, moderazione dell'ingresso di controllo e fisica realizzabilità.

*Nota:* $mu_d$ può essere scelto solo se $mu_s$ non è stato imposto.


== Sintesi del regolatore statico
/*-------------------------RIVEDI----------------------------*/
Il guadagno $mu_s$ e il numero di poli nell'origine in $R_s (s)$ dipende dalla specifica sull'errore a regime $e_infinity$ in risposta a segnali canonici.

Ad esempio, se dobbiamo soddisfare la specifica $|e_infinity| <= e^star$ in risposta ai gradini $w$ e $d$, con $G(s)$ senza poli nell'origine:

possiamo scegliere
$
    R(s) = mu_s >= mu^star
$

oppure
$
    R(s) = mu_s / s
$
nel secondo caso potremo poi scegliere $mu_d$ “liberamente” purché consenta di rispettare i vincoli sull'attenuazione di $d$.



== Sintesi del regolatore dinamico
=== Obiettivi
La progettazione di $R_d (s)$ mira a
#list()
+  imporre $omega_c$ in un certo intervallo;
+  garantire un dato margine di fase $M_f$, cioè $arg(L(j omega_c)) >= -180 + M_f$;
+  garantire una certa attenuazione e pendenza di $L(j omega)$, e anche di $R(j omega)$ a pulsazioni elevate.

Per la *terza specifica* è sufficiente introdurre poli del regolatore a pulsazioni elevate.

Utilizzeremo la sintesi per tentativi individuando dei possibili scenari in base al diagramma di
$
    G_e (s) = R_s (s) G(s)
$
che chiameremo *sistema esteso*.



=== Scenario A
Nell'intervallo (“centrale”) di pulsazioni ammissibili per la pulsazione di attraversamento $omega_c$ esiste un sotto-intervallo in cui la fase di $G_e (j omega)$ rispetta il vincolo sul margine di fase.
#cfigure("Images/Scenario_A.png", 60%)

*Obiettivo:*
- attenuare (selettivamente) il diagramma delle ampiezze (traslarlo in basso) in modo che $omega_c$ ricada nel sotto-intervallo in cui in vincolo sul margine di fase è rispettato;
- alterare meno possibile la fase.

*Azioni possibili:*
- Se $mu_d$ libero, allora scegliere $R_d (s) = mu_d$ con $mu_d < 1$;
- Se $mu_d$ bloccato (vincolato dalla scelta di $mu_s$), allora attenuare mediante inserimento di poli e zeri in $R_d (s)$.

#pagebreak()
==== Caso $mu_d$ libero
#nfigure("mu_d_libero.png", 85%)

==== Caso $mu_d$ vincolato
Per attenuare solo nel range di pulsazioni selezionato progettiamo
#tageq($R_d (s) = frac(1 + alpha tau s, 1+ tau s)$, $0<alpha<1$)
cioè una rete ritardatrice
#nfigure("mu_d_vincolato_1.png", 85%)
#nfigure("mu_d_vincolato_2.png", 85%)

#pagebreak()
=== Rete ritardatrice
#nfigure("Rete_ritardatrice.png", 85%)

==== Tuning approssimato
Il nostro obiettivo è calcolare $alpha$ e $tau$ in modo che $L(j omega)$ abbia una pulsazione di attraversamento $omega_c^star$ e valga $arg(L(j omega_c^star)) approx arg(G_e (j omega_c^star))$.

Procediamo quindi a
#list(tight: false, spacing: 12pt,
[scegliere $alpha$ tale che $20 log alpha approx - |G_e (j omega_c^star)|_"dB"$;],
[scegliere $tau$ tale che $dfrac(1,alpha tau) <= dfrac(omega_c^star, 10)$.]
)


==== Formule di inversione
Dobbiamo calcolare $alpha$ e $tau$ in modo che alla pulsazione $omega_c^star$ (pulsazione a cui vorremmo $|L(j omega)|_"dB" = 0$) la rete ritardatrice abbia una attenuazione $o < M^star < 1$ e uno sfasamento $-frac(pi,2) < phi^star < 0$, ovvero
$
    R_d (j omega_c^star) = M^star e^(j phi^star)
$

Poniamo
$
    &frac(1+j alpha tau omega_c^star, 1+j tau omega_c^star) = M^star (cos phi^star + j sin phi^star)\
    ==>&1+j alpha tau omega_c^star = M^star (cos phi^star + j sin phi^star) (1+j tau omega_c^star)
$
Uguagliano parte reale e parte immaginaria:
$
    1 &= M^star cos phi^star - M^star tau omega_c^star sin phi^star\
    alpha tau omega_c^star &= M^star tau omega_c^star cos phi^star + M^star sin phi^star
$
Arriviamo così alle formule di inversione
$
    tau = frac(cos phi^star - dfrac(1,M^star), omega_c^star sin phi^star)
    space space
    alpha tau = frac(M^star - cos phi^star, omega_c^star sin phi^star)
$
*Nota:* perché si abbia $alpha > 0$ occorre che $M^star < cos phi^star$.

Quindi, noi vogliamo che $|L(j omega)|_"dB" = 0$ per $omega = omega_c^star$. Per farlo scegliamo una $omega_c^star$ e ricaviamo il $M_f^star$ dalle specifiche.\
Calcoliamo quindi $M^star$ e $phi^star$ imponendo
$
    |G_e (j omega_c^star)|_"dB" + 20 log M^star = 0 &space M_f^star = 180 degree + arg(G_e (j omega_c^star)) + phi^star
$
verificando che i risultati trovati soddisfino le relazioni seguenti
$
    0 < M^star < 1 space -frac(pi,2) < phi^star < 0 space M^star < cos phi^star
$
e, infine, calcolare $alpha$ e $tau$ mediante formule di inversione.



=== Scenario B
Nell'intervallo "centrale" di pulsazioni ammissibili per la pulsazione di attraversamento $omega_c$ *NON* esistono pulsazioni in cui la fase di $G_e (j omega)$ rispetta il vincolo sul margine di fase.
#nfigure("Scenario_B.png", 70%)

*Obiettivo:*
- modificare il diagramma delle fasi (aumentare la fase) nell'intervallo in modo che il vincolo sul margine di fase sia rispettato;
- amplificare meno possibile l'ampiezza.

*Azioni possibili:*
- aggiungere uno o più zeri (a pulsazioni precedenti quella di attraversamento desiderata) per aumentare la fase;
- aggiungere uno o più poli a pulsazioni più alte per la fisica realizzabilità e per evitare una eccessiva amplificazione.
Definizione di poli e zeri: @poli_e_zeri[].


==== Aggiunta di uno zero
#nfigure("Scenario_B_aggiunta_zero.png", 70%)

#pagebreak()
==== Aggiunta di due zeri
#nfigure("Scenario_B_aggiunta_2_zeri.png", 70%)


==== Progettazione
Tenendo conto dell'aggiunta di uno o due poli si può progettare $R_d (s)$ come segue.

Si realizza una rete anticipatrice
#tageq($R_d (s) = frac(1 + tau s, 1 + alpha tau s)$, $0< alpha < 1$)

O, nel caso sia necessario un anticipo di fase maggiore, si possono aggiungere due zeri
#tageq($R_d (s) = frac(1 + tau_1 s, 1 + alpha_1 tau_1 s) frac(1 + tau_2 s, 1 + alpha_2 tau_2 s)$, $0< alpha_1 < 1, #h(5pt) 0< alpha_2 < 1$)

Una volta realizzata una rete anticipatrice (singola o multipla) si possono verificare due casi:
#enum(numbering: n => $bold(B_#n)$,
    [$omega_c$ è nell'intervallo di specifica e il vincolo sul margine di fase è rispettato. In questo caso il progetto è terminato;],
    [$omega_c$ è fuori dall'intervallo di specifica o in un intervallo in cui il vincolo sul margine di fase non è rispettato (ci siamo comunque ricondotti ad uno scenario A , quindi esiste un sotto-intervallo in cui il vincolo sul margine di fase è rispettato).]
)

#heading(level: 5, numbering: none)[Caso *$B_2$*]
- Se $mu_d$ libero allora scegliamo $mu_d < 1$ per attenuare
  $
    R_d (s) =  mu_d frac(1+ tau_b s, 1+ alpha_b tau_b s)
  $
- Se $mu_d$ bloccato
  $
    R_d (s) =  mu_d frac(1+ alpha_a tau_a s, 1+ tau_a s) frac(1+ tau_b s, 1+ alpha_b tau_b s)
  $

Quest'ultimo tipo di regolatore viene chiamato *rete ritardo-anticipo*.
#nfigure("Rete_ritardo_anticipo.png", 66%)



==== Rete anticipatrice
#nfigure("Scenario_B_Rete_anticipatrice.png", 88%)

#heading(level: 5, numbering: none)[Formule di inversione]
Calcoliamo $alpha$ e $tau$ in modo che alla pulsazione $omega_c^star$ (pulsazione a cui vorremmo $|L(j omega)|_"dB" = 0$) la rete anticipatrice abbia una
amplificazione $M^star > 1$ e uno sfasamento $0 < phi^star < pi/2$ , ovvero
$
    R_d (j omega_c^star) = M^star e^(j phi^star)
$
Poniamo 
$
    frac(1+j tau omega_c^star, 1+j alpha tau omega_c^star) = M^star (cos phi^star + j sin phi^star)
    space
    1 + j tau omega_c^star = M^* (cos phi^star + j sin phi^star)(1+j alpha tau omega_c^star)
$
Uguagliando parte reale e parte immaginaria
$
    1 &= M^star cos phi^star - M^star alpha tau omega_c^star sin phi^star\
    tau omega_c^star &= M^star alpha tau omega_c^star cos phi^star + M^star sin phi^star
$
Quindi arriviamo alle seguenti formule di inversione
$
    tau = frac(M^star - cos phi^star, omega_c^star sin phi ^star)
    space
    alpha tau = frac(cos phi^star - dfrac(1,M^star), omega_c^star sin phi ^star)
$
*Nota:* perché si abbia $alpha > 0$ occorre che cos $phi^star > dfrac(1,M^star)$.

Come sempre, vogliamo che $|L(j omega)|_"dB" = 0$ per $omega = omega_c^star$, quindi
- scegliamo $omega_c^star$ e ricaviamo $M_f^star$ dalle specifiche
- calcoliamo $M^star$ e $phi^star$ imponendo
  $
    |G_e (j omega_c^star)|_"dB" + 20 log M^star = 0 
    space
    M_f^star = 180 degree + arg(G_e (j omega_c^star)) + phi^star
  $
- verifichiamo che $M^star > 1$, $0 < phi^star < pi , cos phi^star > dfrac(1,M^star)$
- calcoliamo $alpha$ e $tau$ mediante le formule di inversione. 



== Controllori PID
I _controllori PID_, che sta per *controllori ad azione Proporzionale Integrale Derivativa*, sono tra i più usati in ambito industriale. Tra i motivi di questo c'è sicuramente il fatto di poter controllare in modo soddisfacente un'ampia gamma di processi; ma anche perché possono essere usati in casi un cui non vi sia un modello matematico preciso del sistema sotto controllo, perché sono state sviluppati negli anni delle regole per la loro taratura automatica.\
Essi, grazie alla loro semplicità, possono essere realizzati con varie tecnologie, come: meccanica, pneumatica, idraulica, elettronica analogica e digitale, e questo ovviamente implica una grande disponibilità commerciale.

Un PID ideale è rappresentato dalla seguente espressione:
$
    R(s) = K_p (1 + dfrac(1,T_i s) + T_d s)
$
ove $T_i$ è il _tempo integrale_ e $T_d$ il _tempo derivativo_.
#nfigure("PID_1.png", 70%)
L'ingresso di controllo è
$
    U(s) &= R(s) E(s) \
    &= K_p E(s) + frac(K_p,T_i) frac(E(s),s) + K_p T_d s E(s) 
$
che nel dominio del tempo equivale a
$
    u(t) = cal(L)^(-1)  [U(s)] = underbrace(K_p e(t), #text(size: 10pt)[termine \ Proporzionale]) + underbrace(frac(K_p, T_i)integral_0^t e(tau) thick d tau, #text(size: 10pt)[termine \ Integrale]) + underbrace(K_p T_d frac(d e(t), d t),#text(size: 10pt)[termine \ Derivativo])
$

*Attenzione: * il PID ideale non è fisicamente realizzabile. Infatti, sviluppando i calcoli, si vede che la funzione di trasferimento del controllore ha un numeratore con grado più elevato del denominatore:
$
    R(s) 
    &= K_p (1 + dfrac(1,T_i s) + T_d s)\
    &= frac(K_p T_i s + K_p + K_p T_i T_d s^2, T_i s)
$

Il PID “reale” (fisicamente realizzabile) richiede di aggiungere un polo in alta frequenza:
$
    R^("fr") (s) = K_p (1 + dfrac(1,T_i s) + T_d s) frac(1, 1+T_p s)
$

Raccogliendo i termini e definendo opportunamente $tau_1$ , $tau_2$ possiamo vedere che il PID reale è una combinazione di una rete anticipatrice e di una rete ritardatrice:
$
    R^("fr") (s) 
    &= underbrace(frac(K_p, T_i), mu) frac(T_i s + 1 + T_i T_d s^2, s) frac(1, 1+T_p s)\
    &= mu frac((1+tau_1 s)(1+ tau_2 s), s) frac(1, 1+T_p s)
$


=== Casi speciali
*Regolatori P:* se $T_i -> infinity$ e $T_d = 0$, quindi il termine integrale e quello derivativo sono assenti, si ottiene un regolatore proporzionale $R(s) = K_p$.

*Regolatori I:* in assenza di termine proporzionale e derivativo, si ottiene un regolatore puramente integrale $R(s) = frac(K_i, s)$. Si può interpretare come una rete ritardatrice con il polo sposto nell'origine e con lo zero all'infinito.

#par(leading: 1.2em)[
    *Regolatori PI:* se $T_d = 0$, quindi manca il termine derivativo, si ottiene un regolatore proporzionale integrale $R(s) = K_p (1 + dfrac(1, T_i s))$. Possono essere visti come reti ritardatrici con polo nell'origine e zero in $-dfrac(1, T_i)$.
] 

*Regolatori PD:* se $T_i -> infinity$, quindi manca il termine integrale, si ottiene un regolatore proporzionale derivativo $R(s) = K_p (1+T_d s)$. Possono essere visti come reti anticipatrici con zero in $-dfrac(1,T_d)$ e un polo posto all'infinito (nel caso ideale).





















































