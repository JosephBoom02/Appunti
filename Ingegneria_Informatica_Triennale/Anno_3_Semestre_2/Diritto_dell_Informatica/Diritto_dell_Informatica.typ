#let title = "Diritto dell'Informatica"
#let author = "Bumma Giuseppe"

#set document(title: title, author: author)


#show link: set text(rgb("#cc0052"))

#show ref: set text(green)

#set page(margin: (y: 0.5cm))
#set page(margin: (x: 1cm))

#set text(14pt)

#set heading(numbering: "1.1.1.1.1.1")
//#set math.equation(numbering: "(1)")

#set math.mat(gap: 1em)

//Code to have bigger fraction in inline math 
#let dfrac(x,y) = math.display(math.frac(x,y))

//Equation without numbering (obsolete)
#let nonum(eq) = math.equation(block: true, numbering: none, eq)
//Usage: #nonum($a^2 + b^2 = c^2$)

#let space = h(5em)

//Color
#let myblue = rgb(155, 165, 255)
#let myred = rgb(248, 136, 136)

//Shortcut for centered figure with image
#let cfigure(img, wth) = figure(image(img, width: wth))
//Usage: #cfigure("Images/Es_Rettilineo.png", 70%)

#let nfigure(img, wth) = figure(image("Images/"+img, width: wth))

#set highlight(extent: 2pt)


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

// colors
#let mygray = rgb(242, 242, 242)

#outline()

= Concetti giuridici di base

#rect(fill: mygray, stroke: 1pt)[Che cos'è una norma giuridica? E una sentenza?]
La norma giuridica è un comando o precetto generale ed astratto, alla base dell'ordinamento giuridico,
che impone o proibisce un certo comportamento. La sentenza costituisce la giurisprudenza, il frutto
dell'attività del giudice.
#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[I precedenti giurisprudenziali sono vincolanti in Italia? E nei paesi di common law?]
No, ma possono essere presi come punto di riferimento. Invece sì.
#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Fonti del diritto]

Regolamenti europei, costituzione, leggi ordinare (d.l. d.lgs.), leggi regionali, regolamenti, usi e
consuetudini.
#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Regolamenti comunitari e direttive comunitarie]

I primi vengono subito attuati, sono già leggi dello Stato. Le direttive devono essere recepite tramite d.lgs.
#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Differenza tra legge, decreto legge e decreto legislativo]

La prima deve essere approvata dal Parlamento ed emanata dal PdR. Il d.l. viene emanato dal governo in
atti di urgenza e necessità. Il d.lgs. è emanato dal governo con legge delega.
#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Se è stata pubblicata una direttiva ed è scaduto il termine per il recepimento nei singoli ordinamenti
giuridici, posso applicare direttamente la direttiva anche in assenza di una legge nazionale?]

Sì se è sufficientemente chiara.
#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Che cosa succedese un articolo di un contratto o un regolamento di un'amministrazione locale viola
una legge nazionale? Sono validi?]

No.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Efficacia della legge nel tempo: in quale momento una legge inizia ad avere efficacia? E
quando cessa di avere efficacia?]

Se non è stabilito dalla legge stessa, entra in vigore dopo 15 giorni dalla pubblicazione sulla gazzetta
ufficiale. Cessa quando viene abrogata (esplicita se viene dichiarata incostituzionale o abrogata, oppure
implicita quando i contenuti di una nuova legge superanola vecchia legge).

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Efficacia della legge nello spazio]

Una legge nazionale è valida in Italia, il regolamento in UE, la legge regionale in una regione.
#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Quali sono i principali criteri di interpretazione della legge?]

L'interpretazione autentica viene data da chi ha emanato la legge, oppure abbiamo l'interpretazione
giurisprudenziale, da un giudice. Oppure dagli esperti del diritto, la dottrina.
#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Persona fisica e persona giuridica]

Le società costituite per atto pubblico hanno personalità giuridica, sono titolari di diritti e di doveri (come
le società o le associazioni). La persona fisica è la persona umana titolare di diritti e di doveri.
#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Differenza tra capacità giuridica e capacità di agire]

La capacità giuridica è l'insieme dei diritti che si acquisisconosin dalla nascita (vita, nome, salute, integrità
fisica, famiglia). La capacità di agire si acquisisce con la maggiore età ed è la capacitàdi compiere atti
giuridici, cioè atti rilevanti per l'ordinamento giuridico (comprare una casa, comprare una macchina).
#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Diritti reali]

Diritti che si hanno su un bene, come la proprietà, l'usufrutto, il diritto di abitazione.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Obbligazioni e fonti delle obbligazioni]

Le obbligazioni nascono da un contratto, da un fatto illecito o da ogni altro atto o fatto idoneo a produrlo.
Le fonti sono il contratto o il fatto illecito. L'obbligazione è un diritto a una prestazione personale (soggetto
attivo creditore, soggetto passivo debitore).

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Requisiti del contratto]

Accordo delle parti (il contratto è concluso quando al proponente arriva l'accettazione), causa (deve
essere lecita), oggetto (possibile, lecito, determinato, determinabile), forma (quando è prescritta dalla
legge sotto pena di nullità).

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Nullità e annullabilità del contratto]

Il contratto è nullo quando è contrario a norme imperative, cioè quando è contrario all'ordinamento
giuridico. Oppure la mancanza dei requisiti all'articolo 1325 (accordo, causa, oggetto, forma) e l'illiceità
dei motivi (oggetto o causa illeciti). Invece l'annullabilità si ha quando una delle parti era legalmente
incapace di contrattare, quando manca la capacità di agire. Il contratto non è annullabile se il minore ha
raggirato la sua minore età, ma potrebbe non essere sufficiente. È annullabile per i vizi del consenso, cioè
errore, violenza e dolo (inganno).

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Contratti tipici e atipici]

I primi sono quelli previsti dall'ordinamento giuridico. Quelli atipici da un altro ordinamento.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Il contratto, in generale, deve essere stipulato per iscritto e firmato? E' valido un contratto
verbale? Qualiproblemi potrebbero sorgere da un contratto non scritto?]

Un contratto verbalepuò essere valido, a meno che la legge non preveda la forma scritta con pena di
nullità.


= Diritto d'autore

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Che cosa tutela la legge sul diritto d'autore?]

Le opere dell'ingegno di carattere creativo (qualunque modo o forma di espressione), programmi per
l'elaboratore come opere letterarie, banche dati (creazione intellettuale).

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Chi è titolare del diritto d'autore?]

Il creatore dell'opera.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Quando sorge il diritto d'autore?]

Al momento della creazione.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Quanto dura il diritto d'autore?]

Per i diritti patrimoniali fino a 70 anni dalla morte dell'autore. Per le opere collettive è 70 anni dalla prima
pubblicazione. Per le amministrazioni è 20 anni dalla prima pubblicazione. I diritti morali sono esercitabili
dai discendenti e ascendenti o coniuge.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Quali sono i diritti morali d'autore?]

Diritto di rivendicare la paternità dell'opera, integrità dell'opera (opporsi a modificazione dell'opera che
crea pregiudizi all'onore e alla reputazione dell'autore), diritto di inedito (decidere se e quando pubblicare
l'opera, se l'autore aveva deciso di non pubblicare l'opera, i suoi eredi non potranno farlo) e diritto di
pentimento (ritirare l'opera o derivate dal commercio per gravi ragioni morali, con l'obbligo di rimborso
verso chi ha acquistato i diritti di utilizzazione economica, è personale e intrasmissibile).

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Quali sono i diritti patrimoniali d'autore?]

Dirittidi utilizzazione economica (pubblicazione, riproduzione, distribuzione, elaborazione, noleggio,
prestito).

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[I diritti patrimoniali devono essere trasferiti tutti insieme o possono essere ceduti separatamente?]

Sono indipendenti, quindi possono essere ceduti separatamente.


#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Diritto di riproduzione, di modificazione, di distribuzione]

Il primo è la moltiplicazionein copie in qualunque forma, eccezion fatta per riproduzioni senza rilievo
economico e utilizzo legittimo. Il secondo è la trasformazione dell'opera che costituisceun rifacimento
sostanziale dell'opera.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Principio di esaurimento comunitario del software]

Il diritto di distribuzione dell'originale si esaurisce in CE se la prima vendita o il primo atto di trasferimento
è effettuato dal titolare o con il suo consenso. Non si applica alla messa a disposizione on demand.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Che cosa sono le libere utilizzazioni?]

Ad esempio una citazione di un'opera protetta da diritto d'autore ma con un fine didattico. L'uso lecito.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Che cosa sono i diritti connessi?]

Sono diritti patrimoniali che spettano a chi permette la fruizione dell'opera.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Come è tutelato il software nell'ordinamento giuridico italiano?]

Il software è protetto da diritto d'autore, in qualsiasi forma (sorgente o eseguibile), ma deve essere
originale. Non sono protette le idee e i principi. È tutelatala forma espressiva (struttura e sviluppo delle
istruzioni che compongono il programma).

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Chi è titolare dei diritti patrimoniali sul software? E dei diritti morali?]

Autore, altri soggetti, il dipendente.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Differenza tra opera collettiva e opera in comunione]

La prima è la riunione di opere di autori distinti che vengono messe insieme da un altro soggetto che
coordina il lavoro nel suo complesso e diviene a sua volta autore, con diritti diversi da quelli dagli altri sulle
opere. La seconda è un'opera in cui il contributo degli autori è indistinguibile, il diritto d'autore appartiene
a tutti gli autori, in tutte le parti che sono uguali, salvo accordo scritto.


#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Cosa succede se il dipendente di un'azienda sviluppa un software? Di chi sono i diritti morali
e patrimoniali?]

Il datore di lavoro è titolare del diritto esclusivo di utilizzazione economica sul software se esso è
sviluppato nello svolgimento delle sue mansioni o su istruzioni del datore di lavoro, salvo patto contrario.
Il diritto morale è sicuramente del lavoratore.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[In che cosa consistono i diritti esclusivi sul software?]

Riproduzione, modificazione, distribuzione, il principio di esaurimento.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Quali sono le eccezioni per consentire l'interoperabilità?]

Se la riproduzione e traduzione del codice sono indispensabili per ottenere le informazioni necessarie per
l'interoperabilità. Lecito se le attività sono compiute dal licenziatario o soggetto autorizzato, se le
informazioni non sono facilmente accessibili, se le attività sono limitate alle parti del software necessarie.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Se sviluppate un software, è necessario registrarlo? E' utile? Perché?]

È facoltativa. Può essere utile per fornire una prova dell'esistenza del software e della titolarità dei diritti.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Brevettabilità del software]

La concessione dei brevetti software è esclusa, è tutelato dal diritto d'autore, salvo il caso in cui siano parte
integrante dell'invenzione.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Misure tecnologiche a protezione del software]

Tecnologie, dispositivi, componenti aventi lo scopo di impedire o limitare atti non autorizzatidai titolari
dei diritti (es. accesso, copia). Le possono apporre i titolari dei diritti d'autore, diritti connessi, costitutori
di banche dati. Sono controlli di accesso, protezione o limitazione.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Le principali sanzioni (civili, penali, amministrative) a tutela delle opere dell'ingegno]

- *Sanzioni civili*: lesione del diritto di utilizzazione economica, con risarcimento e distruzione o rimozione dello stato di fatto. 
- *Sanzioni penali*: multe per la riproduzione, diffusione di un'opera altrui, messa a disposizione del pubblico, riproduzione di un numero maggiore di esemplari di quelle di cui aveva diritto, duplicazione per profitto. 
- *Sanzioni amministrative*: il doppio del prezzo di mercato dell'opera per ogni violazione e per ogni esemplare abusivamente duplicato o riprodotto.


= Banche di dati e opere multimediali

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Quali banche di dati possono essere protette in base alla normativa sul diritto d'autore? Tutte
o no? Quali caratteristiche devono avere per essere protette?]

Sono protette dal diritto d'autore le banche di dati che per la scelta o la disposizione del materiale
costituiscono una creazione intellettuale dell'autore.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Definizione di banca di dati]

Raccolta di opere, dati o altri elementi ndipendenti sistematicamente o metodicamente disposti ed
individualmente accessibili mediante mezzi elettronici o in altro modo.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Che cosa è tutelato della banca di dati?]

La forma espressiva (struttura, disposizione, scelta dei contenuti). Non si estende al contenuto della base
dei dati. Non si estende al software per la costituzione e il funzionamento della banca di dati.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Il concetto di creatività/originalità]

Per originalità si intende originalità intrinseca nel contenuto. Se non sono originali i dati, la base di dati
può essere considerata originale se la disposizione è originale.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[L'elenco telefonico è una banca di dati tutelabile? Perché?]

No secondo una sentenza, mancano di originalità e creatività.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Banca di dati selettiva e non selettiva]

La prima ha un contenuto scelto discrezionalmente dall'autore, che ne costituisce l'originalità. La seconda
include tutti i dati possibili sull'argomento senza selezione e l'originalità va cercata nella struttura.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Quando sorge il diritto d'autore su una banca di dati?]

Quando vi è la creazione dell'opera. Spesso è un'opera collettiva o in comunione.


#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Quali sono i diritti esclusivi dell'autore di una banca di dati?]

Riproduzione, traduzione, distribuzione al pubblico dell'originale o di copie, presentazione,
dimostrazione, comunicazione in pubblico.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Quali sono le deroghe al diritto d'autore su di una banca di dati?]

Accesso o consultazione per scopi didattici, di ricerca scientifica, non svolta nell'ambito di un'impresa.
Bisogna indicare la fonte. La riproduzione non è permanente. Oppure impiego per scopi di sicurezza
pubblica o per effetto di una procedura amministrativa o giurisdizionale.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Uso da parte dell'utente legittimo. Che cosa può fare?]

Riproduzione, presentazione, distribuzione, modificazione se necessario per il suo normale impiego.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[In che cosa consiste il diritto del costitutore?]

Si dice anche diritto sui generis, è un diritto indipendente e parallelo al diritto d'autore, ha la natura di
diritto connesso. Si possono vietare operazioni di estrazione e rimpiego dell'intera base dati, anche di parti
non sostanziali. Sono vietate se contrarie alla normale gestione o arrecano pregiudizio al costitutore.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Quanto dura il diritto del costitutore?]

Dura 15 anni dal momento del completamento della base di dati. Per quelle messe al pubblico 15 anni dal
1° gennaio dell'anno successivo della messa al pubblico.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Perché il legislatore ha riconosciuto il diritto del costitutore?]


#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Il costitutore deve dare un apporto creativo?]

No.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Vi sono eccezioni e limitazioni anche al diritto del costitutore?]

Sì, le eccezioni che si applicano al diritto d'autore, le libere utilizzazioni.


#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Principio di esaurimento comunitario delle banche di dati]

La prima vendita di una base di dati o di una copia esaurisce il diritto di controllare la vendita successiva,
come il dirittod'autore. Se la base di dati è trasmessa online però è una prestazione di servizi.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Che cosa succede se vengono effettuati nuovi investimenti su una banca di dati?]

Decorre un nuovo termine di tutela del diritto del costitutore, 15 anni dal completamento della base di
dati modificata o dalla sua messa a disposizione del pubblico.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Se intende creare un sito web, quali problemi si pone sotto il profilo della proprietà
intellettuale? Vi sono altri aspetti legali da considerare? (V. obblighi informativi commercio
elettronico, privacy, ecc.)]

Non è semplice capire se un sito web possa essere tutelato dal diritto d'autore, può contenere opere
dell'ingegno creative protette dal diritto d'autore. La dottrina dice che se il sito web è creativo e originale
può essere considerato opera dell'ingegno.


= Contratti a oggetto informatico

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Il concetto di contratto a oggetto informatico]

Il contratto a oggetto informatico è la licenza software, cioè il licenziante cede al licenziatario il diritto di
godimento del software e la documentazione accessoria per il tempo stabilito. È un contratto atipico. Non
si cede la titolarità del programma o lo sfruttamento economico.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Che cos'è un contratto?]

È l'accordo di due o più parti di costituire, regolare o estinguer un rapporto giuridico patrimoniale.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Il principio di autonomia contrattuale]

Si può determinare liberamente il contenuto di un contratto a patto che sia a norma di legge.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Quali sono i requisiti del contratto?]

L'accordo, la causa, l'oggetto, la forma (può essere prescritta dalla legge a pena di nullità).

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Quando è concluso un contratto?]

Quando la proposta ha conoscenza dell'accettazione dell'altra parte. Si può accettare l'accettazione
tardiva se se ne dà avviso all'altra parte. Può essere imposta una forma di accettazione. Se il contratto si
esegue senza una preventiva risposta, è concluso all'inizio dell'esecuzione, l'accettante deve avvisare
prima dell'esecuzione (pena risarcimento del danno). Oppure la proposta può essere revocata, ma se
l'accettante ha intrapreso in buona fese l'esecuzione, il proponente deve risarcire il danno.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[I contratti devono essere stipulati per iscritto/firmati per la loro validità?]

Dipende, se la legge detta una forma vincolata, sono validi sono in quella forma. È concessa la forma
verbale.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Quali sono le potenziali criticità di un contratto concluso solo verbalmente?]

Laddove dovesse presentarsi una contestazione, in un contratto scritto si ha la certezza dell'accordo,
verbalmente no.


#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Cosa sono le condizioni generali di contratto?]

Sono predisposte dai contraenti. Sono efficaci nei confronti dell'altro.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Alcune clausole delle condizioni generali di contratto devono essere specificamente approvate
per iscritto? Perché? Può farmi qualche esempio di clausola c.d. vessatoria? Per quale motivo è
necessaria una doppia sottoscrizione?]

Non hanno effetto se non approvate per iscritto le limitazioni di responsabilità, la facoltà di recedere, di
sospendere l'esecuzione o sanciscono per l'altro le decadenze, le limitazioni alle eccezioni, restrizioni coi
terzi, tacita proroga o rinnovazione, clausole compromissorie e deroghe. Le clausole vessatorie
determinano uno squilibrio di diritti e obblighi tra le parti. Ad esempio limitare le azioni o i diritti del
consumatore in caso di inadempimento da parte del professionista o imporre il pagamento di una somma
di denaro eccessiva nel caso di ritardo di un adempimento. Si applica la doppia sottoscrizione perché con
la prima l'aderente accetta il contenuto delle condizioni generali di contratto non onerose, con la seconda
da apporsi in modo specifico il contenuto di quelle vessatorie.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Come si può gestire la doppia sottoscrizione nei contratti online?]

Tramite la firma digitale.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Contratti conclusi mediante moduli o formulari.]

Le clausole aggiunte a mano prevalgono sulle clausole del modulo a favore della parte opposta.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Clausole vessatorie nei contratti con i consumatori.]

C'è uno squilibrio di diritti ed obblighi a sfavore del consumatore. L'accettazione deve essere fatta
autonomamente e separatamente, altrimenti non hanno effetto.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Contratto di licenza d'uso di software]

Licenziante cede al licenziatario il diritto di godimento del sw e la documentazione accessoria per il tempo
stabilito. Può essere previsto un corrispettivo. Se la licenza è gratuita si parla di freeware. NON si cede la
titolarità del programma, non si cede il diritto di sfruttamento economico. Si tratta di un contratto atipico.
Possono essere applicate le norme del codice civile sulla locazione per quantocompatibili e salvo accordo
contrario.


#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Se per sviluppare un software Le occorre utilizzare un altro software/libreria scaricato da
Internet mediante licenza, a quali clausole della licenza deve prestare attenzione?]

Esclusività, trasferibilità, proprietà intellettuale, riservatezza, utilizzo per scopi personali, numero
massimo di installazioni.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Garanzie e responsabilità per vizi. Rimedi per l'utilizzatore]

Di solito ci sono clausole che escludono la responsabilità. Se il software non permette di ottenere risultati
utili, si può restituire o riparare, o sostituire o ottenere un indennizzo.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Licenza "a strappo"]

Il software è confezionato in un involucro trasparente che indica le condizioni del contratto: se aperto si
accetta il contratto.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Contratto di sviluppo software]

Una parte si obbliga a studiare, sviluppare e realizzare un software in base alle richieste dell'altra parte. Se
è un imprenditore è un contratto d'appalto di servizi. Se è un professionista è un contratto di prestazione
d'opera intellettuale.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Se dovessero proporLe un contratto di sviluppo software, a quali clausole presterebbe
particolare attenzione e perché?]

Conformità del software alle specifiche, il tempo e il luogo, l'utilizzo di macchine, parti da sviluppare
separatamente, collaudo, costo, diritti di proprietà intellettuale, responsabilità dei danni.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Appalto e contratto d'opera intellettuale]

Nel primo caso, una parte assume con organizzazione dei mezzi e gestione a proprio rischio il compimento
di un'opera per conto di un'altra parte, c'è l'obbligo di risultato, divieto di subappalto senza autorizzazione
del committente. Nel secondo caso, l'obbligo viene assunto da un professionista e non ha un obbligo di
risultato, ma un obbligo di mezzi.


= Proprietà industriale

#rect(fill: mygray, stroke: 1pt)[Che cosa tutela la proprietà industriale?]

Marchi, indicazioni geografiche, disegni, modelli, invenzioni, topografie dei prodotti a semiconduttori,
segreti commerciali, nuove varietà vegetali.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Quando sorgono e come si acquisiscono i diritti di proprietà industriale?]

Mediante brevettazione per le invenzioni e registrazione per i marchi.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[I marchi oggetto di registrazione]

Tutti i segni, le parole i nomi di persone, i disegni, le lettere, le cifre, i suoni, la forma del prodotto o della
confezione purché servanoa distinguere i prodotti o servizi e ad essere rappresentati nel registro in modo
tale da consentire alle autorità di determinare chiaramente l'oggetto della protezione. Un marchio si può
registrare per una o più classi.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Quali segni non possono essere registrati come marchi?]

I segni simili per prodotti o servizi identici o affini, o distintivi, conta anche la notorietà.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Il requisito di novità]

Non si possono registrare marchi che in base alla notorietà di un marchio già esistente può portare a
confusione.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Quali segni non possono essere oggetto di tutela per mancanza di capacità distintiva?]

I segni divenuti di uso comune nel linguaggio corrente o usi costanti nel commercio. Denominazione
generiche, salvo i segni che a seguito dell'uso e prima della domanda di registrazione abbiano acquistato
carattere distintivo.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Chi può registrare un marchio?]

Chiunque, ma hanno diritto chi utilizza per il commercio o prestazione di servizi, non chi presenta
domanda in mala fede, le amministrazioni dello Stato.


#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Quali sono i diritti del titolare di un marchio registrato?]

Uso esclusivo, diritto di vietare ai terzi l'uso di un segno identico o simile per prodotti identici o affini, anche
non affini se il marchio ha rinomanza.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Quali sono le limitazioni al diritto di marchio?]

Non si può vietare l'uso nell'attività economica, purché valgano i principi della correttezza professionale.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Il trasferimento del marchio]

Il marchio può essere trasferito o concesso in licenza per la totalità o una parte dei prodotti o servizi per i
quali è stato registrato, salvo la violazione delle condizioni di licenza da parte del licenziatario.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Quando/come sorgono i diritti esclusivi su di un'invenzione industriale?]

Con il brevetto.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Il brevetto per invenzione industriale]

Sono brevettabili le invenzioni di ogni settore della tecnica nuove che implicano un'attività inventiva e
sono destinate all'applicazione industriale. È escluso per le scoperte, teorie, piani, principi, programmi in
quanto tali (sono brevettabili solo se danno un risultato tecnico).

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Il requisito di novità delle invenzioni]

Nuovo vuol dire non compreso allo stato della tecnica.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Il concetto di priorità]

Quando si fa una domanda di deposito di brevetto ha 12 mesi di tempo per decidere in quali paesi far
valere ilbrevetto.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Il concetto di attività inventiva]

Un'invenzione implica un'attività inventiva se per una persona esperta essa non risulti evidentemente allo
stato della tecnica.


#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Il concetto di industrialità]

Si tratta di applicazione industriale se un'invenzione può essere fabbricata o utilizzata in qualsiasi
industria, compresa quella agricola.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Come si fa domanda di brevetto?]

Online sul sito di UIBM, presso qualunque Camera di Commercio, tramite posta all'UIBM.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Cosa sono le rivendicazioni?]

Si intende ciò che debba formare oggetto del brevetto. La descrizione e i disegni servono ad interpretare
le rivendicazioni, per avere un'equa protezione al titolare e una sicurezza giuridica ai terzi.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Quali sono gli effetti della brevettazione?]

Decorrono dalla pubblicazione della domanda, 18 mesi dopo il deposito o 90 giorni se il richiedente ha
dichiarato di volerla pubblicare immediatamente.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Qual è la durata di un brevetto?]

Stessi diritti del brevetto italiano a decorrenza dalla data di pubblicazione sul Bollettino europeo.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[In che cosa consistono i diritti morali? E quelli patrimoniali?]

I primi sono di essere riconosciuti autori e alla morte da discendenti o ascendenti. I secondi sono alienabili
e trasmissibili.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Le invenzioni dei dipendenti]

Nel primo caso idiritti spettano al datore di lavoro se l'attività inventiva è posta come oggetto del contratto
e retribuita. Se non è retribuita e l'invenzione è fatta in adempimento di un contratto, i diritti sono del
datore di lavoro, ma spetta al lavoratore un equo premio. Altrimenti se l'invenzione rientra nel campo di
attività del datore di lavoro, esso ha il diritto di opzione per l'uso, l'acquisto del brevetto, la facoltà di
chiedere brevetti all'estero, e dovrà pagare un canone al lavoratore; ha diritto di opzione per 3 mesi dalla
comunicazione del deposito della domanda di brevetto. L'invenzione del dipendente è tale se fatta
durante l'esecuzione del contratto o entro un anno da quando l'inventore ha lasciato l'azienda o
amministrazione.


#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[La nullità del brevetto]

Il brevetto è nullo se l'invenzione non è brevettabile, se non è descritta in modo sufficientemente chiaro,
se l'oggetto si estende oltre il contenuto della domanda iniziale, se il titolare non aveva diritto di ottenerlo.
La nullità può essere parziale. Ilbrevetto nullo può produrre gli effetti di un diverso brevetto. Il titolare del
brevetto convertito può entro 6 mesi presentare una domanda di correzione del brevetto. Se il brevetto
prevede un prolungarsi dei tempi chi ha investito per sfruttare la brevettabilità dopo il vecchio termine ha
diritto a ottenere una licenza obbligatoria e gratuita per il periodo di maggior durata. Ha effetto retroattivo
ma non pregiudica atti eseguiti o contratti eseguiti e pagamenti a titolo di equo premio, canone o prezzo.


= Protezione dei dati personali

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Chi ha diritto alla protezione dei dati personali?]

Le persone fisiche, non quelle giuridiche.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[A quali trattamenti di dati si applica il Regolamento sulla protezione dei dati persona?]

Si applica al trattamento di dati personali automatizzati o non automatizzati in archivio. Non si applica
alle persone fisiche in attività personali o domestiche e ad autorità competenti a fini di prevenzione di
reati.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Ambito di applicazione territoriale - Ambito di applicazione materiale]

Il primo dice che il regolamento si applica a titolari o responsabili stabiliti nell'UE o non stabiliti nell'UE se
riguarda dati di interessati che si trovano nell'UE per l'offerta di beni o prestazione di servizi o per
monitorare il loro comportamento all'interno dell'UE.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Se tratto dati personali per scopi esclusivamente personali sono soggetto al Regolamento?]

No.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Che cos'è un dato personale?]

Qualsiasi informazione di una persona fisica identificata o identificabile direttamente o indirettamento,
come il nome, ID, ubicazione, codice fiscale. Si distinguono in comuni e sensibili.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Che cosa si intende per trattamento? Esempi di attività che costituiscono un trattamento di dati personali]

Qualsiasi operazione o insieme di operazioni compiute con o senza ausilio di processi automatizzati per la
raccolta, registrazione, conservazione, uso, consultazione, modifica dei dati personali. Degli esempi sono
la profilazione e la pseudonimizzazione.


#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Se memorizzate il cellulare di un vostro compagno di corso che incontrate in giro dovete fornirgli
l'informativa privacy? Si/no? Perché?]

No, perché un uso personale.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[L'indirizzo IP/un nickname sono dati personali? Perché?]

Sì, perché identificano indirettamente una persona fisica.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Il Regolamento si applica ai dati anonimi? Quand'è che un dato è anonimo?]

Il dato anonimo non si sa a chi può essere riferita, ma può diventare dato personale se collegato a
informazioni di diversa natura e risulti idoneo a identificare un soggetto.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Cosa significa anonimizzare e pseudonimizzare un dato?]

Anonimizzare un dato vuol dire rendere il dato non riconducibile alla persona. Pseudonimizzare il dato
vuol dire rendere un dato personale riconducibile alla persona esclusivamente solo se associati ad altri
dati.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Chi sono i soggetti coinvolti nel trattamento? (= Titolare, Responsabile, Interessato, Soggetto
designato, DPO)]

Il titolare determina le finalità e i mezzi del trattamento. Il responsabile tratta i dati personali per conto del
titolare. L'interessato è la persona fisica a cui si riferiscono i dati personali. Il DPO svolge numerosi compiti
di supporto all'applicazione del regolamento.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Come si fa a nominare un responsabile del trattamento? Posso farlo a voce?]

Sempre tramite atto scritto con le generalità, i compiti, le funzioni.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Secondo quali principi fondamentali deve essere effettuato un trattamento?]

Liceità, correttezza e trasparenza, limitazione della finalità, minimizzazione dei dati, esattezza,
limitazione della conservazione, integrità e riservatezza, responsabilizzazione.


#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Liceità del trattamento (6 condizioni)]

Almeno una delle condizioni deve essere soddisfatta. 
+ L'interessato ha espresso il consenso. 
+ Il trattamento è necessario all'esecuzione di un contratto o le misure precontrattuali. 
+ Il trattamento è necessario per adempiere un obbligo legale. 
+ Il trattamento è necessario per la salvaguardia degli interessi vitali dell'interessato o di un'altra persona fisica. 
+ Il trattamento è necessario per l'esecuzione di un compito di interesse pubblico o connesso. 
+ Il trattamento è necessario per legittimo interesse del titolare.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Trattamento di categorie particolari di dati personali]

Non si devono trattare dati personali su razza o etnia, politica, religione, filosofia, appartenenza sindacale,
genetica, biometrica, salute, vita sessuale. Sono i dati sensibili. Non vale se l'interessato ha prestato il
consenso esplicito per finalità specifiche o se è necessario per assolvere obblighi sul diritto al lavoro,
protezione sociale, se è necessario in sede giudiziaria, se è di interesse pubblico rilevante.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Se qualcuno intende trattare i Suoi dati personali, a quali adempimenti è tenuto per rispettare
la normativa?]

Deve attuare misure tecniche adeguate, come la pseudonimizzazione quando deve determinare i mezzi di
trattamento e all'atto del trattamento stesso.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Per quanto tempo posso conservare dei dati personali?]

Per un tempo non superiore al conseguimento delle finalità per cui sono trattati. Altrimenti di più a fini di
archiviazione nel pubblico interesse.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Che cos'è l'informativa? Chi la deve fornire? Quale contenuto ha (esempi)? Ha un contenuto
obbligatorio minimo? Quando deve essere fornita? Può non essere fornita?]

L'informativa deve contenere l'identità del titolare, i dati di contato del DPO, le finalità e la base giuridica,
i destinatari dei dati, l'intenzione di trasferirli a un paese terzo, il periodo di conservazione, i diritti
dell'interessato, il diritto di revoca, il diritto di reclamo, se è un requisito necessario, la notifica diun
processo decisionale automatizzato come la profilazione. La fornisce il titolare del trattamento. È un
obbligo del GDPR.


#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Quali sono i diritti dell'interessato?]

Accesso, rettifica, integrazione, cancellazione (oblio), limitazione del trattamento, portabilità dei dati,
opposizione.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Il consenso. Deve sempre essere richiesto per poter trattare dati personali a norma di legge?
Quali caratteristiche deve avere? Se l'interessato tace, vale come consenso?]

Non sempre, è necessario per quanto riguarda il trattamento di dati sensibili, ma non serve per
investigazioni o in ambito della regolare attività d'impresa. No se non si riesce a dimostrare che
l'interessato abbia dato il consenso.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Vi sono casi in cui non occorre il consenso? Quali (esempi)?]

Sì, per esempio per tutelare la salute o il diritto alla vita dell'interessato.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Obblighi e responsabilità del Titolare: misure tecniche e organizzative]

Lo deve fare per proteggere i dati con la minimizzazione, garantire il soddisfacimento dei requisiti del
regolamento e tutelare i diritti degli interessati.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Articolo 25 del GDPR : Protezione dei dati fin dalla progettazione e protezione dei dati per
impostazione predefinita]
1. Tenendo conto dello stato dell'arte e dei costi di attuazione, nonché della natura, dell'ambito di applicazione, del contesto e delle finalità del trattamento, come anche dei rischi aventi probabilità e gravità diverse per i diritti e le libertà delle persone fisiche costituiti dal trattamento, sia al momento di determinare i mezzi del trattamento sia all'atto del trattamento stesso il titolare del trattamento mette in atto misure tecniche e organizzative adeguate, quali la pseudonimizzazione, volte ad attuare in modo efficace i principi di protezione dei dati, quali la minimizzazione, e a integrare nel trattamento le necessarie garanzie al fine di soddisfare i requisiti del presente regolamento e tutelare i diritti degli interessati.
2. Il titolare del trattamento mette in atto misure tecniche e organizzative adeguate per garantire che siano trattati, per impostazione predefinita, solo i dati personali necessari per ogni specifica finalità del trattamento. Tale obbligo vale per la quantità dei dati personali raccolti, la portata del trattamento, il periodo di conservazione e l'accessibilità. In particolare, dette misure garantiscono che, per impostazione predefinita, non siano resi accessibili dati personali a un numero indefinito di persone fisiche senza l'intervento della persona fisica.
3. Un meccanismo di certificazione approvato ai sensi dell'articolo 42 può essere utilizzato come elemento per dimostrare la conformità ai requisiti di cui ai paragrafi 1 e 2 del presente articolo.



#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Responsabile del trattamento: spiegare il ruolo e le responsabilità]

Tratta i dati per conto del titolare, fornisce garanzie sufficienti per mettere in atto misure tecniche e
organizzative adeguate. Può nominare un subresponsabile con autorizzazione del titolare. Questo è il DP,
non il DPO.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Registri delle attività di trattamento del Titolare e del Responsabile: cosa sono, chi è tenuto a
predisporli, in quale forma? Cosa devono contenere? Perché sono importanti?]

Li devono tenere tutte le imprese con più di 250 dipendenti; le imprese con meno di 250 dipendenti devono
tenerlo se il trattamento rappresenta un rischio per i diritti e le libertà dell'individuo, o non sia un
trattamento occasionale o includa il trattamento di particolari categorie di dati. Va tenuto in forma scritta,
anche elettronica. Il registro del titolare deve contenere nome e dati di contatto del titolare e del DPO, le
finalità del trattamento, la descrizione delle categorie di interessati e delle categorie dei dati personali, le
categorie di destinatari, trasferimenti verso paesi terzi, termini ultimi per la cancellazione di dati,
descrizione generale delle misure di sicurezza tecniche. Il registro del responsabile deve contenere nome
e dati di contatto del responsabile, di ogni titolare per cui agisce e del DPO, le categorie dei trattamenti, i trasferimenti verso paese terzo, l'identificazione del paese terzo, descrizione delle misure di sicurezza
tecniche.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Sicurezza del trattamento - misure di sicurezza tecniche e organizzative]

Il titolare deve garantire e dimostrare che il trattamento è effettuato in modo conforme al regolamento.
Le misure sono riesaminate e aggiornate. Si attuano politiche adeguate all'attività di trattamento, come il
codice di condotta o un meccanismo di certificazione. Si tiene conto di perdita o accesso illegale ai dati
trattati. Degli esempi sono pseudonimizzazione, cifratura, capacità di assicurare riservatezza, integrità, di
ripristinare l'accesso, di testare l'efficacia delle misure.


#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Articolo 24 del GDPR: Responsabilità del titolare del trattamento]

1. Tenuto conto della natura, dell'ambito di applicazione, del contesto e delle finalità del trattamento, nonché dei rischi aventi probabilità e gravità diverse per i diritti e le libertà delle persone fisiche, il titolare del trattamento mette in atto misure tecniche e organizzative adeguate per garantire, ed essere in grado di dimostrare, che il trattamento è effettuato conformemente al presente regolamento. Dette misure sono riesaminate e aggiornate qualora necessario.
2. Se ciò è proporzionato rispetto alle attività di trattamento, le misure di cui al paragrafo 1 includono l'attuazione di politiche adeguate in materia di protezione dei dati da parte del titolare del trattamento.
3. L'adesione ai codici di condotta di cui all'articolo 40 o a un meccanismo di certificazione di cui all'articolo 42 può essere utilizzata come elemento per dimostrare il rispetto degli obblighi del titolare del trattamento.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Notifica violazioni dei dati (data breach): in che cosa consiste? Cosa si deve fare?]

Il titolare deve notificare la violazione al Garante privacy entro 72 ore a meno che la violazione non presenti
un rischio per i diritti e le libertàdelle persone fisiche. Il responsabile deve informare il titolare.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Comunicazione violazioni dei dati (data breach): in che cosa consiste? Cosa si deve fare?]

Se la violazione dei dati presenta un rischio elevato per i diritti e le libertà delle persone fisiche il titolare
deve comunicare la violazione all'interessato. A meno che il titolare abbia attuato misure di protezione
tecniche e organizzative, le abbia adottate successivamente per scongiurare un rischio elevato o se la
comunicazione richiederebbe sforzi sproporzionati, si procede con una comunicazione pubblica in
quest'ultimo caso.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Valutazione d'impatto sulla protezione dei dati]

Se un tipo di trattamento ha un rischio elevato per i diritti e le libertà delle persone fisiche, prevedendo
l'uso di nuove tecnologie, il titolare deve effettuare prima di procedere al trattamento una valutazione
dell'impatto dei trattamenti previsti sulla protezione dei dati personali, una singola valutazione può
esaminare trattamenti simili con rischi analoghi.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Responsabile della Protezione dei Dati (DPO): quando deve essere nominato, ruolo, compiti]

Deve essere nominato da tutte le aziende del settore pubblico. Nel settore privato va nominato quando il
titolare effettua trattamenti su larga scala di categorie particolaridi dati personali. Deve informare e
fornire consulenza al titolare e ai dipendenti del trattamento, sorvegliare l'osservanza del regolamento,
fornire pareri sulla valutazione d'impatto, cooperare con l'autorità di controllo.


#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Trasferimenti di dati extra UE: a quali condizioni sono consentiti?]

Decisioni di adeguatezza, garanzie adeguate, per esempio norme vincolanti d'impresa, clausole
contrattuali simili al regolamento. In mancanza delle precedenti, con il consenso dell'interessato, se
necessario per conclusione o esecuzione di un contratto, giurisprudenza, interesse pubblico, interessi
vitali dell'interessato o di altre persone o se il trasferimento avviene da un registro pubblico.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[A quali condizioni possono essere inviati SMS, e-mail pubblicitarie?]

È possibile se si può dimostrare di aver ottenuto il consenso esplicito e sistematico dell'interessato.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Qual è l'attuale disciplina per le telefonate pubblicitarie? Cosa può fare per tutelarsi? Il Registro
pubblico delle opposizioni?]

Diventano reato le telefonate pubblicitarie se si registra il numero al Registro pubblico delle opposizioni.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Se un'impresa ha un sito web (es. di e-commerce), quali sono i principali problemi di privacy
che deve affrontare?]

Se tratta dati deve mostrare in chiaro l'informativa della privacy policy con i tipi di dati che vengono
trattati, i destinatari, come vengono trattati.



= Firme elettroniche e documenti digitali

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Che cosa si intende per documento?]

Possiamo inquadrarlo come scritture private, riproduzioni meccaniche, atti pubblici, non c'è una
definizione precisa nell'ordinamento.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Scrittura privata]

Fa piena prova fino a querela di falso, la firma non deve essere disconosciuta.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Atto pubblico]

È un documento redatto da un notaio o altro pubblico ufficiale autorizzato, dà certezza ufficiale e può
essere contestato. Degli esempi sono l'atto di compravendita immobiliare o l'atto di costituzione di società
di capitali.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Riproduzioni meccaniche]

Ogni rappresentazione meccanica di fatti e cose (fotografie, fonogrammi, informatica, cinematografica)
costituiscono piena prova se colui contro il quale sono prodotte non né disconosce la conformità.

#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Concetto di firma elettronica, firma elettronica avanzata, qualificata e digitale. Definizione ed effetti giuridici]

La firma elettronica sono dati in forma elettronica acclusi o connessi ad altri dati elettronici usati per
firmare. La firma avanzata è una firma elettronica connessa unicamente al firmatario, idonea a identificare
il firmatario, la può usare il firmatario sotto il proprio esclusivo controllo, è collegata ai dati sottoscritti. La
firma qualificata è avanzata e creata da un dispositivo per la creazione di una firma qualificata e basata su
un certificato qualificato. A una firma elettronica non sono negati gli effetti giuridici, la firma qualificata ha
effetti di firma autografa, se rilasciata in UE è qualificata in UE. La firma digitale è una firma qualificata
basata su crittografia asimmetrica. Le firme qualificate o digitali sono valide se è associabile un riferimento
temporale opponibile.


#line(length: 100%)
#rect(fill: mygray, stroke: 1pt)[Che cos'è un certificato qualificato? Quali sono le principali informazioni che contiene?]

È un attestato elettronico che convalida la firma elettronica, è rilasciato da un fornitore di servizi. Contiene
il nome o pseudonimo del firmatario, il periodo di validità, la firma elettronica avanzata o sigillo del
prestatore di servizi fiduciari qualificato che rilascia il certificato.




