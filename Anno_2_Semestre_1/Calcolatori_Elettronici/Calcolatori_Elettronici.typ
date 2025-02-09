
#import "@preview/physica:0.9.0": *
#import "@preview/i-figured:0.2.3"
#import "@preview/cetz:0.1.2" 
#import "@preview/xarrow:0.2.0": xarrow


#import cetz.plot 

#let title = "Calcolatori Elettronici T"
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

= Guida agli esami
== Mapping
=== Attivare una data sezione di memoria
La consegna ci dice che è presente un'ulteriore memoria `EPROM`, denominata `EPROM_OPT`, da 1 GB
mappata a partire da `0x40000000` in accordo a quanto indicato in seguito.
All'avvio, `EPROM_OPT` dovrà essere disabilitata. Se non abilitata, `EPROM_OPT`
dovrà abilitarsi in seguito alla lettura consecutiva di quattro byte divisibili per
8 (zero escluso) da `INPUT_PORT` mentre, quando abilitata, `EPROM_OPT` dovrà
disattivarsi in seguito alla lettura consecutiva da `INPUT_PORT` di quattro byte
divisibili per 8 (zero escluso). Il procedimento di abilitazione e disabilitazione di
`EPROM_OPT` dovrà avvenire continuamente mediante opportune reti logiche e
senza alcun ausilio software.

I chip select di `EPROM_OPT` saranno:
```
CS_EPROM_OPT_0 = BA31*·BA30·BE0·ACTIVE
CS_EPROM_OPT_1 = BA31*·BA30·BE1·ACTIVE
CS_EPROM_OPT_2 = BA31*·BA30·BE2·ACTIVE
CS_EPROM_OPT_3 = BA31*·BA30·BE3·ACTIVE
```

Mentre le reti logiche necessarie sono le seguenti:
#cfigure("Images/2025-02-08-15-30-42.png", 90%)






== Chip Select
=== Come funziona la lettura da un chip select
Abbiamo
- un chip select CS_STATUS mappato a `0x80000002h`
- `BD[15..0]` già utilizzato
- necessità di leggere un segnale `INT_INPUT_SYNC`

#cfigure("Images/2025-02-08-15-07-44.png", 80%)
Quando `CS_STATUS` e `MEMRD` sono asseriti, il driver 3-state permette il passaggio dei dati.
Quando viene effettuata una lettura all'indirizzo di `CS_STATUS`
- i bit `BD[23..17]` vengono impostati a 0
- il bit `BD[16]` viene impostato al valore corrente di `INT_INPUT_SYNC` 


Ora prendiamo come riferimento questo schema:
```
Indirizzo: 0x80000002 (ultimi due bit: 10)
Word:      BD[7..0] | BD[15..8] | BD[23..16] | BD[31..24]
Byte #:     Byte 0  |  Byte 1   |   Byte 2   |  Byte 3
Indirizzo:    00        01           10          11
```
se si esegue una LBU andando a leggere su `CS_STATUS` (`0x80000002h`) si leggono i bit `BD[23..16]` che ha come bit meno significativo il valore di `INT_INPUT_SYNC`, mentre gli altri bit a 0.






== Lettura e scrittura dalle porte
=== Scrivere i dati in input letti da INPUT_PORT a un dato indirizzo
Riporto l'esempio dell'esame del 21/12/2023 in cui viene chiesto che #highlight(fill: red)[il dato _signed_ letto da *INPUT_0* dovrà essere scritto a *FFFFFFF0h*], mentre #highlight(fill: purple)[i dati _unsigned_ letti da *INPUT_1* dovranno essere scritti a *80000000h*].

In questo caso le porte trasferiscono sullo stesso bus dati perché non trasferiscono mai in contemporanea.
#cfigure("Images/2025-02-08-10-57-24.png", 80%)
#cfigure("Images/2025-02-08-15-17-05.png", 80%)
Codice del DLX:
```yasm
00000000: LHI R20,0x6000      ; R20 = 60000000h
00000004: LBU R21,0x0001(R20) ; legge il valore di ACTIVE_0
00000008: BEQZ R21,PORT_1     ; se R21=0 è attiva INPUT_PORT_1
;----------------------------------------------------------------------------
;Questo è il codice per leggere da PORT_0
0000000C: SUBI R22,R0,0x0010  ; R22 = FFFFFFF0h, quindi carico in un registro l'indirizzo in cui bisogna scrivere il dato
00000010: LB R21,0x0000(R20)  ; legge byte signed da INPUT_PORT_0
00000014: SB R21,0x0000(R22)  ; scrive il byte letto a FFFFFFF0h
00000018: RFE
;----------------------------------------------------------------------------
;Questo è il codice per leggere da PORT_1
PORT_1:                       ; porta attiva, INPUT_PORT_1
0000001C: LHI R22,0x8000      ; R22 = 80000000h
00000020: LBU R21,0x0000(R20) ; legge byte unsigned da INPUT_PORT_1
00000024: SB R21,0x0000(R22)  ; scrive il byte letto a 80000000h
00000028: RFE
```

=== Scrivere un dato letto da un un idirizzo in OUTPUT_PORT
La consegna dice che in `OUTPUT_PORT` dovrà essere scritto il byte letto all'indirizzo `0xFF000020`
#cfigure("Images/2025-02-08-15-23-51.png", 80%)
Codice DLX dell'interrupt handler
```yasm
00000000: LHI R20,0x8000 ; R20 = 80000000h
00000004: LHI R24,0xFF00 ; R24 = FF000000h
00000008: LBU R21,0x0002(R20) ; legge il valore di INT_INPUT_SYNC
0000000C: BEQZ R21,OUTPUT ; se R21=0, gestisce OUTPUT_PORT
00000010: LBU R21,0x0000(R20) ; legge byte da INPUT_PORT
00000014: SB R21,0x0000(R24) ; scrive il byte letto a FF000000h
00000018: RFE
;--------------------------------------------------------------
;Questa è la sezione di codice che ci interessa
OUTPUT: ; scrive un byte in OUTPUT_PORT
0000001C: LBU R21,0x0020(R24) ; legge byte a FF0000020h
00000020: SB R21,0x0001(R20) ; scrive byte in OUTPUT_PORT
00000024: RFE
```


=== Leggere un dato da `INPUT_PORT` e scriverlo in `OUTPUT_PORT`
La consegna di dice che nel sistema sono presenti  due porte in input, `INPUT_0` e `INPUT_1` già progettate, ciascuna in grado di trasferire 8 bit mediante il protocollo di
handhsake. Mediante le due porte in input dovranno essere eseguiti *unicamente trasferimenti di dati a 16 bit* appena questo si rende possibile.

Inoltre, nel sistema è presente una porta in output, denominata `OUTPUT_PORT`,
in grado di trasferire 8 bit di dato verso l'esterno mediante il protocollo di
handshake. Il trasferimento verso `OUTPUT_PORT` dovrà avvenire, quando
possibile, unicamente e contemporaneamente alla lettura dei dati a 16 bit da
`INPUT_0` e `INPUT_1` quando un segnale proveniente dall'esterno denominato
`TRANSFER_OUT` risulta asserito. In queste circostanze, il valore da trasferire
verso l'esterno mediante `OUTPUT_PORT` coincide con il dato letto da `INPUT_1`.
I dati a 16 bit letti da `INPUT_0` e `INPUT_1` dovranno essere scritti a `B0000000h`

#cfigure("Images/2025-02-08-21-57-07.png", 90%)
In `OUTPUT_PORT` deve essere traferito il dato letto da `INPUT_PORT_1`.

Al fine di poter eseguire un trasferimento verso la porta in output
durante le letture dalle due porte in input, è necessario che:
- la porta in output sia pronta a eseguire un trasferimento, quindi deve essere asserito il segnale `INT_OUTPUT`
- sia asserito il segnale `TRANSFER_OUT` proveniente dall'esterno

È importante però sincronizzare opportunamente questi due segnali, in particolare essi necessitano di un doppio campionamento:
#cfigure("Images/2025-02-08-22-02-00.png", 80%)
#cfigure("Images/2025-02-08-22-02-31.png", 70%)
Utilizzando due flip-flop in cascata (come mostrato nello schema con i due FFD), si garantisce che il segnale campionato sia stabile prima di essere utilizzato dal resto del sistema. Il #highlight(fill: blue)[primo FFD cattura il segnale], mentre il #highlight(fill: red)[secondo assicura che eventuali oscillazioni o glitch siano stati eliminati].

#cfigure("Images/2025-02-09-10-42-07.png",90%)

*N.B.* `CS_FREEZE` è un segnale indispendabile in questi casi:
- Impedisce che variazioni transitorie dei segnali `TRANSFER_OUT` e `INT_OUTPUT` possano propagarsi nel sistema quando non desiderato
- Quando `CS_FREEZE` è attivo (1):
  - Il multiplexer seleziona l'ingresso 1 (`TRANSFER_OUT` o `INT_OUTPUT`)
  - Permette il campionamento di un nuovo valore
- Quando `CS_FREEZE` è inattivo (0):
  - Il multiplexer seleziona l'ingresso 0 (`TRANSFER_OUT_SYNC` o `INT_OUTPUT_SYNC`)
  - Mantiene il valore precedentemente campionato

Questo verrà utilizzato nel codice del DLX Interrupt prima di una lettura: si fa una lettura fittizia (_dummy read_) all'indirizzo in cui è mappato `CS_FREEZE`, cosicché le variazioni dei segnali `TRANSFER_OUT` o `INT_OUTPUT` vengano propagate correttamente sulle uscite dei FFD, cioè sui segnali `TRANSFER_OUT_ENABLED` e `OUTPUT_READY`.

Codice DLX dell'interrupt handler:
```yasm
00000000 LHI R25,0x4000       ; R25=40000000h
00000004 LBU R26,0x0002(R25)  ; CS_FREEZE (dummy read)
00000008 LHI R27,0xB000       ; R27=B0000000h
0000000C LHU R26,0x0000(R25)  ; legge 16 bit da porte in input
00000010 SH R26,0x0000(R27)   ; scrive 16 bit a B0000000h
00000014 RFE
```
Come si evince dal codice il trasferimento dei dati da `INPUT_PORT_1` a `OUTPUT_PORT` non avviene via software, ma solo con l'ausilio di reti combinatorie.



=== Trasferimenti da diverse porte di Input
Riferimento all'esame del 17/01/2023.

Progettare un sistema, basato su un processore DLX nel quale sono presenti quattro porte in input (denominate `INPUT_PORT_0`, `INPUT_PORT_1`, `INPUT_PORT_2` e `INPUT_PORT_3`) e una porta in output (denominata
`OUTPUT_PORT`).\
Sin dall'avvio, e mediante l'ausilio di opportune reti logiche: *ogni 6
trasferimenti di un byte di tipo #highlight(fill: yellow)[_signed_] da `INPUT_PORT_0`, dovrà essere
eseguito un unico trasferimento* (a 32 bit) *dalle 4 porte in input* e così via
(i.e., 6 trasferimenti da `INPUT_PORT_0`, un unico trasferimento a 32 bit dalle
quattro porte, 6 trasferimenti da `INPUT_PORT_0`, eccetera). Inoltre, il byte letto
da `INPUT_0` (indipendentemente dal fatto che si stia leggendo da una singola o
dalle quattro porte in input) dovrà essere contemporaneamente inviato, nel
caso questo sia possibile, anche a `OUTPUT_PORT`.\
Quanto letto dalla/e porta/e in input dovrà essere scritto, come word,
all'indirizzo `0xF0000008`.

I Chip Select delle porte saranno:
```yasm
CS_PORT 60000000h
  CS_INPUT_PORT_0 60000000h (CS_PORT + 0)
  CS_INPUT_PORT_1 60000001h (CS_PORT + 1)
  CS_INPUT_PORT_2 60000002h (CS_PORT + 2)
  CS_INPUT_PORT_3 60000003h (CS_PORT + 3)
```
#cfigure("Images/2025-02-09-11-44-36.png", 90%)
#cfigure("Images/2025-02-09-11-45-00.png", 90%)


Un contatore modulo 8 consente di tenere traccia dei trasferimenti
dalla/dalle porta/e in input in accordo a quanto indicato nel testo
del problema. In particolare, il segnale `32_BIT` asserito indica che il
trasferimento dovrà essere effettuato contemporaneamente dalle 4 porte
in input. Tale segnale, ottenuto elaborando l'uscita del contatore,
risulta:
#cfigure("Images/2025-02-09-12-26-10.png",90%)

Il segnale `32_BIT` è utilizzato anche per condizionare la richiesta di
interrupt inviata al `DLX` nel modo seguente:
```yasm
INT_DLX = INT_INPUT_0·32_BIT* +
          INT_INPUT_0·INT_INPUT_1·INT_INPUT_2·INT_INPUT_3·32_BIT
```

Per velocizzare l'esecuzione dell'interrupt handler, si evita di
verificare quale tipo di trasferimento è abilitato. Pertanto, non si
legge il segnale `32_BIT` che indica se il trasferimento deve avvenire
unicamente da `INPUT_PORT_0` o contemporaneamente dalle quattro porte in
input.
A tal fine, con l'ausilio della rete seguente, il codice
dell'interrupt handler eseguirà sempre una lettura di una word anche
quando è necessario trasferire solo da `INPUT_PORT_0` (i.e., quando
`32_BIT=0`) evitando così la lettura via software di `32_BIT` e una
consecutiva istruzione di branch.
#cfigure("Images/2025-02-09-12-28-49.png", 70%)
*N.B.* Siccome il byte da leggere da `INPUT_PORT_0` è di tipo _signed_, non possiamo semplicemente mettere i bit `BD[31..8]` a 0, ma bisogna seguire la regola dell'esetensione del segno:
- se il numero è positivo, si pongono a 0 i bit rimanenti
- se il numero è negativo, si pongono a 1 i bit rimanenti
ed è per questo che #highlight(fill: yellow)[colleghiamo a `BD[15..8]`, `BD[23..16]`, e `BD[31..24]` il bit `(BD7)^8`] (che indica il segno del numero), e non semplicemente degli 0.

Infine, nel sistema è anche presente una porta in output attraverso la
quale trasferire, quando possibile, il dato letto da `INPUT_PORT_0`
contemporaneamente all'esecuzione di questa operazione.

#cfigure("Images/2025-02-09-12-35-16.png", 70%)
Il segnale #text(fill: red)[`OUTPUT_ENABLED`], utilizzato per condizionare il chip-select
di `OUTPUT_PORT_0`, è ottenuto campionando sul fronte di salita di `MEMRD`
il segnale `INT_OUTPUT_0` come segue:
#cfigure("Images/2025-02-09-12-34-49.png", 70%)

Codice `DLX` dell'interrupt handler:

```yasm
00000000: LHI R20,0x6000      ; R20 = 60000000h
00000004: LW R21,0x0000(R20)  ; legge in R21 una word a 60000000h
00000008: LHI R22,0xF000      ; R22 = F0000000h
0000000C: SW R21,0x0008(R22)  ; scrive R21 a F0000008h
00000010: RFE
```



















