
#import "@preview/physica:0.9.0": *
#import "@preview/i-figured:0.2.3"
#import "@preview/cetz:0.1.2" 
#import "@preview/xarrow:0.2.0": xarrow


#import cetz.plot 

#let title = "Ingegneria del Software"
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

#set text(15pt)

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


#outline()

//Justify text

= Diagrammi UML
== Premessa 
- In questo blocco vedremo i principali Diagrammi UML per la fasi di Analisi del Problema e del Progetto
  
	- per ora non vedremo i diagrammi di deployment e dei componenti
- In particolare:
  
	- Diagramma dei Package e Diagramma delle Classi → parte “*statica*” dell'Architettura Logica e dell'Architettura del Sistema
	  
		- definiscono le entità di sistema senza analizzare come possano interagire tra loro
	- Diagramma di Sequenza → parte “*interazione*” dell'Architettura Logica e dell'Architettura del Sistema
	  
		- definiscono come le entità, definite nella parte precedente, interagiscono tra loro
	- Diagramma di Stato e Diagramma delle Attività → parte “*comportamentale*” dell'Architettura Logica e dell'Architettura del Sistema
	  
		- definiscono il comportamento delle entità del sistema
- Per ogni Diagramma vedremo solo i concetti fondamentali
- In caso di dubbi fare riferimento alla specifica UML 2.5.1
== Unified Modeling Language
- È un *linguaggio* che serve per visualizzare, specificare, costruire, documentare un sistema e gli elaborati prodotti durante il suo sviluppo
- Ha una semantica e una notazione standard, basate su un metamodello integrato, che definisce i costrutti forniti dal linguaggio
- La notazione (e la semantica) è estensibile e personalizzabile
- È utilizzabile per la modellazione durante tutto il ciclo di vita del software (dai requisiti al testing) e per piattaforme e domini diversi
- Combina approcci di:
  
	- modellazione dati (Entity/Relationship)
	- business Modeling (workflow)
	- modellazione a oggetti
	- modellazione di componenti
- Prevede una serie di diagrammi standard, che mostrano differenti viste architetturali del modello di un sistema
- UML è un linguaggio e non un processo di sviluppo
- UML propone un ricco insieme di elementi a livello utente; tuttavia è alquanto informale sul modo di utilizzare al meglio i vari elementi
  
	- ciò implica che per comprendere un diagramma un lettore deve conoscere il contesto in cui esso è collocato
== Diagrammi di UML 2.5.1
- Diagrammi di struttura:
  
	- diagramma delle classi (class)
	- diagramma delle strutture composite (composite structure)
	- diagramma dei componenti (component)
	- diagramma di deployment (deployment)
	- diagramma dei package (package)
	- diagramma dei profili (profile)
- Diagrammi di comportamento:
	- diagramma dei casi d'uso (use case)
	- diagramma di stato (state machine)
	- diagramma delle attività (activity)
	- diagrammi di interazione:
		- diagramma di comunicazione (communication)
		- diagramma dei tempi (timing)
		- diagramma di sintesi delle interazioni (interaction overview)
		- diagramma di sequenza (sequence)
#cfigure("images/image_1710162664687_0.png", 90%)
== UML 2.5.1 e Visio
- Visio 2016 gestisce nativamente solo alcuni dei Diagrammi di UML 2.5.1
	- classi
	- attività
	- sequenza
	- stato
	- casi d'uso
- È possibile però scaricare uno stencil per poter gestire anche gli altri diagrammi: http://softwarestencils.com/uml/index.html
== Diagramma dei Package
==== Package
- Un package, in programmazione, è un contenitore di classi
- Un package, in generale, è utilizzato per raggruppare elementi e fornire loro un namespace
- Un package può essere innestato in altri package
- NAMESPACE: è una porzione del modello nella quale possono essere definiti e usati dei nomi
- In un namespace ogni nome ha un significato univoco
#cfigure("images/image_1710162829556_0.png", 90%)
==== Diagramma dei Package
- Un diagramma dei package è un diagramma che illustra come gli elementi di modellazione sono organizzati in package e le relazioni (dipendenze) tra i package
#cfigure("images/image_1710162883736_0.png", 90%)
- Un diagramma può rappresentare package con diversi livelli di astrazione
==== Dipendenze
- UML permette di rappresentare relazioni che NON sussistono fra istanze nel dominio rappresentato, ma sussistono fra gli elementi del modello UML stesso o fra le astrazioni che tali elementi rappresentano
- Dipendenza:
  è rappresentata da una linea tratteggiata orientata che va da un elemento dipendente (Source) ad uno indipendente(Target)
#cfigure("images/image_1710163145240_0.png", 80%)
- Una dipendenza indica che cambiamenti dell'elemento *indipendente* influenzano l'elemento *dipendente*
  
	- modificano il significato dell'elemento dipendente
	- causano la necessità di modificare anche l'elemento dipendente perché i significati sono dipendenti
- UML mette a disposizione nove diversi tipi di dipendenza, ma per i nostri fini consideriamo quasi sempre <<use>>
#cfigure("images/image_1710163162642_0.png", 50%)
==== Diagramma dei Package
- Tutti i diagrammi che utilizzeremo nella fase di analisi saranno trasformati per venir utilizzati nella fase di progettazione (in analisi definisco il "cosa" e in progettazione definisco il "come")
- Quando si usa il diagramma dei package per definire la parte strutturale dell'architettura logica ricordare sempre che si stanno esprimendo *dipendenze logiche* che sussistono tra le entità del problema
- Non è detto che tali dipendenze rimangano tali anche nella fase di progettazione
- Esempio: in sistema per l'interpretazione di una grammatica si hanno due parti fondamentali
  
	- Lexer→strumento che legge e spezza una sequenza di caratteri in sotto-sequenze dette “token”
	- Parser prende in ingresso i token generati, li analizza e li elabora in base ai costrutti specificati nella grammatica stessa
==== Esempio
#cfigure("images/image_1710163246985_0.png", 90%)
È ovvio che è il Parser a usare il Lexer, quindi il diagramma prodotto nella fase di analisi sarà così
#cfigure("images/image_1710163288049_0.png", 90%)
Ma nel diagramma del progetto è il Lexer a fare il Parser
#cfigure("images/image_1710174515137_0.png", 90%)

== Interfaccia
- Le interfacce forniscono un modo per partizionare e caratterizzare gruppi di proprietà
- Un'interfaccia non deve specificare come possa essere implementata, ma semplicemente quello che è necessario per poterla realizzare
- Le entità che realizzano l'interfaccia dovranno fornire una “*^^vista pubblica^^*”(attributi, operazioni, comportamento osservabile all'esterno) conforme all'interfaccia stessa
- Se un'interfaccia dichiara un attributo, non significa necessariamente che l'elemento che realizza l'interfaccia debba avere quell'attributo nella sua implementazione, ma solamente che esso apparirà così a un osservatore esterno
==== Notazione
#cfigure("images/image_1710163566742_0.png", 90%)
  Il diagramma sopra e il diagramma sotto hanno lo stesso significato

== Diagramma delle Classi
- Un diagramma delle classi descrive il tipo degli oggetti facenti parte di un sistema e le varie tipologie di relazioni statiche tra di essi
- I diagrammi delle classi mostrano anche le proprietà e le operazioni di una classe e i vincoli che si applicano alla classe e alle relazioni tra classi
- Le proprietà rappresentano le caratteristiche strutturali di una classe:
	- sono un unico concetto, rappresentato però con due notazioni molto diverse: attributi e associazioni
	- benché il loro aspetto grafico sia molto differente, concettualmente sono la stessa cosa
==== Esempio
#cfigure("images/image_1710163858337_0.png", 100%)
- La molteplicità da OrdinePizza a Cliente è *uno a molti*: un cliente può avere da $1$ a $n$ ordini
- La molteplicità da Cliente a OrdinePizza è *univoca*: un ordine è relativo a uno e un solo cliente
- Associazione di *composizione* tra Pizza e OrdinePizza
- Associazione *navigabile* da Pizza e PizzaStandard: da Pizza ci interessa capire qual è la PizzaStandard, ma da PizzaStandard non ci interessa andare a Pizza, cioè ci interessa sapere ogni Pizza da quale PizzaStandard deriva
- Associazione di *aggregazione* tra PizzaStandard e Ingredienti
- Ogni Pizza può avere delle modifiche di ingredienti
- *Classe di associazione* Modifica, che aggiunge attributi e/o comportamenti a quell'associazione
	- Per ogni Pizza, ogni Ingrediente è presente in una certa quantità, che può essere modificata
==== Classe
- Una *classe* modella un insieme di entità (le istanze della classe) aventi tutti lo stesso tipo di caratteristiche (attributi, associazioni, operazioni...)
- Ogni classe è descritta da:
	- un nome
	- un insieme di caratteristiche (feature): attributi, operazioni, ...
#cfigure("images/image_1710163911503_0.png", 90%)
=== Attributi
- La notazione degli attributi descrive una proprietà con una riga di testo all'interno del box della classe
- La forma completa è:
	- `visibilità nome:tipo molteplicità==default {stringa di proprietà}`
- Un esempio di attributo è:
	- `stringa: String[10] == “Pippo”{readOnly}`
- L'unico elemento necessario è il nome
	- Visibilità: attributo pubblico (+), privato (-) o protected( == )
	- Nome: corrisponde al nome dell'attributo
	- Tipo: vincolo sugli oggetti che possono rappresentare l'attributo
	- Default: valore dell'attributo in un oggetto appena creato
	- Stringa di proprietà: caratteristiche aggiuntive (readOnly)
	- Molteplicità: ...
==== Molteplicità
- È l'indicazione di quanti oggetti possono entrare a far parte di una proprietà
- Le molteplicità più comuni sono:
	- `1, 0..1, *`
- In modo più generale, le molteplicità si possono definire indicando gli estremi inferiore e superiore di un intervallo (per esempio `2..4`).
- Molti termini si riferiscono alla molteplicità degli attributi:
	- Opzionale: indica un limite inferiore di 0
	- Obbligatorio: implica un limite inferiore di 1 o più
	- A un solo valore: implica un limite superiore di 1
	- A più valori: implica che il limite superiore sia maggiore di 1, solitamente 
==== Visibilità
- È possibile etichettare ogni operazione o attributo con un identificatore di visibilità
- UML fornisce comunque quattro abbreviazioni per indicare la visibilità:
	- + (public)
	- \- (private)
	- \~ (package)
	- \# (protected)
==== Attributi: Molteplicità
- Nelle istanze, il valore di un attributo multi-valore si indica mediante un insieme
#cfigure("images/image_1710168708671_0.png", 90%)
==== Operazioni
- Le operazioni sono le azioni che la classe sa eseguire, e in genere si fanno corrispondere direttamente ai metodi della corrispondente classe a livello implementativo
- Le operazioni che manipolano le proprietà della classe di solito si possono dedurre, per cui non sono incluse nel diagramma
- La sintassi UML completa delle operazioni è
  `visibilità nome (lista parametri) : tipo ritorno {stringa di propr}`
	- Visibilità: operazione pubblica (+) o privata (-)
	- Nome: stringa
	- Lista parametri: lista parametri dell'operazione
	- Tipo di ritorno: tipo di valore restituito dall'operazione, se esiste
	- Stringa di proprietà: caratteristiche aggiuntive che si applicano all'operazione
==== Operazioni e Attributi Statici
- UML chiama *static* un'operazione o un attributo che si applicano a una classe anziché alle sue istanze
- Questa definizione equivale a quella dei membri statici nei linguaggi come per esempio java e C
- Le caratteristiche statiche vanno sottolineate sul diagramma
#cfigure("images/image_1710169509255_0.png", 60%)
==== Associazioni
- Le associazioni sono un altro modo di rappresentare le proprietà
- Gran parte dell'informazione che può essere associata a un attributo si applica anche alle associazioni
- Un'associazione è una linea continua che collega due classi, orientata dalla classe sorgente a quella destinazione
- Il nome e la molteplicità vanno indicati vicino all'estremità finale dell'associazione:
- la classe destinazione corrisponde al tipo della proprietà
  
  #cfigure("images/image_1710169568202_0.png", 70%)

- È possibile assegnare un nome all'associazione e anche assegnare dei nomi ai “*ruoli*” svolti da ciascun elemento di un associazione
- Gran parte delle cose ce valgono per gli attributi valgono anche per le associazioni
- Anche nei casi in cui non è strettamente necessario, il ruolo può essere utile per aumentare la leggibilità del diagramma
- Ovviamente quando abbiamo un associazione il tipo della proprietà corrisponde alle classi che sono dall'altra parte della associazione, cioè se guardiamo a la proprietà nella classe A sarà di tipo classe B, e viceversa se guardiamo a classe B la proprietà sarà di tipo classe A

#cfigure("images/image_1710169694503_0.png", 95%)


==== Associazioni Bidirezionali
- Una tipologia di associazione è quella *#text(blue)[bidirezionale (o binaria)]*, costituita da una coppia di proprietà collegate, delle quali una è l'inversa dell'altra
- Il collegamento inverso implica che, se seguite il valore di una proprietà e poi il valore della proprietà collegata, dovreste ritornare all'interno di un insieme che contiene il vostro punto di partenza
  
	#cfigure("images/image_1710169720862_0.png", 75%) 

  la natura bidirezionale dell'associazione è palesata dalle *frecce di navigabilità* aggiunte a entrambi i capi della linea
- Con questa specifica stiamo dicendo all'implementatore che nel progetto deve essere garantita la navigabilità in entrambi i sensi; nel caso non ci sia una specifica di navigabilità la realizzazione è deliberata all'implementatore


==== Associazioni Ternarie
Quando si hanno associazioni ternarie (o che coinvolgono più classi) si introduce il simbolo “#text(blue)[diamante]”
  #cfigure("images/image_1710169756579_0.png", 80%)


==== Associazioni: molteplicità
- La specifica UML (vista fino a ora) dichiara che la molteplicità di un'associazione è
  
  #align(center)[#text(blue)[_the multiplicity of instances of that entity is the range of number of objects that participate in the association from the perspective of the other end (the other class)_]]
	in italiano
	#align(center)[#text(blue)[_la molteplicità delle istanze di quell'entità è l'intervallo del numero di oggetti che partecipano all'associazione dal punto di vista dell'altro estremo (l'altra classe)_]]

- Tale definizione (derivata dalla specifica E/R originale) non può però applicarsi alle associazioni multiple
- Pertanto, come già visto nel corso SIT, la notazione usata in questo corso (e in altri) prevede che
  #align(center)[#text(blue)[_la molteplicità di un'associazione rappresenti il numero (minimo e massimo) di istanze dell'associazione a cui un'istanza dell'entità può partecipare_]]
=== Classi di Associazione
- In UML non si possono aggiungere attributi come in ER, ma si creano delle classi di associazione, che permettono di aggiungere attributi, operazioni e altre caratteristiche a una classe, che sono proprie dell'associazione
- In UML si indicano le classi di associazione con una linea tratteggiata, o utilizzando un rombo, come in figura
  
  #cfigure("images/image_1710169956471_0.png", 100%)

- In generale le classi di associazione sono delle classi che sono venute fuori più avanti nel progetto, perché, ad esempio, l'associazione in figura si può realizzare anche con una classe normale
- La classe di associazione aggiunge implicitamente un vincolo extra: ci può essere #underline[solo un'istanza della classe di associazione] tra ogni coppia di oggetti associati



=== Aggregazione e Composizione
- #highlight(fill: myblue)[Aggregazione:]
	- è un associazione che corrisponde a una relazione intuitiva Intero-Parte (“*part-of*”)
	- è rappresentata da un *#text(blue)[diamante vuoto]* sull'associazione, posto vicino alla classe le cui istanze sono gli “interi”
	  #cfigure("images/image_1710170826307_0.png", 40%)
	- è una relazione *binaria*
	- può essere *ricorsiva*
- #highlight(fill: myblue)[Composizione:]
	- è un aggregazione che rispetta due vincoli ulteriori:
		- una parte può essere inclusa in al massimo un intero in ogni istante
		- solo l'oggetto intero può creare e distruggere le sue parti, cioè, le parti non esistono al di fuori dell'intero; questo implica, nell'implementazione, che non esiste un costruttore per le parti
	- è rappresentata da un *#text(blue)[diamante pieno]* vicino alla classe che rappresenta gli “interi”
	  #cfigure("images/image_1710170843379_0.png", 40%)
	- se l'oggetto che compone viene distrutto,anche i figli vengono distrutti, anche se i figli possono essere creati/distrutti in momenti diversi dalla creazione/distruzione dell'oggetto che compone
	- può essere *ricorsiva*
=== Generalizzazione
- La generalizzazione è indicata con una *freccia vuota* fra due classi dette *#text(blue)[sottoclasse]* e *#text(blue)[superclasse]*
- Il significato della generalizzazione è il seguente:
  ogni istanza della sottoclasse è anche istanza della superclasse
  #cfigure("images/image_1710171110210_0.png", 40%)
- La stessa superclasse può partecipare a diverse generalizzazioni
- Una generalizzazione può essere *disgiunta*, cioè le sottoclassi sono disgiunte (non hanno istanze in comune), o meno
  #cfigure("images/image_1710171249961_0.png", 70%)
- Una generalizzazione può essere *#text(blue)[completa]* (l'unione delle istanze delle sottoclassi è uguale all'insieme delle istanze della superclasse), o meno
- *Attenzione*: i valori di default sono `{incomplete, disjoint}`
  #cfigure("images/image_1710171327443_0.png", 75%)
- In una generalizzazione la sottoclasse non solo può avere caratteristiche aggiuntive rispetto alla superclasse, ma può anche *#text(blue)[sovrascrivere]* (*#text(blue)[overriding]*) le proprietà ereditate dalla superclasse
  #cfigure("images/image_1710171353810_0.png", 70%)
=== Generalizzazione Multipla
- Con la *#text(blue)[generalizzazione singola]* un oggetto appartiene a un solo tipo, che può eventualmente ereditare dai suoi tipi padre
- Con la *#text(blue)[generalizzazione multipla]* un oggetto può essere descritto da più tipi, non necessariamente collegati dall'ereditarietà
- Si noti che la generalizzazione multipla è una cosa ben diversa dall'ereditarietà multipla
	- #highlight(fill: myred)[Ereditarietà multipla:] un tipo può avere più supertipi, ma ogni oggetto deve sempre appartenere a un suo tipo ben definito
	- #highlight(fill: myred)[Generalizzazione multipla:] un oggetto viene associato a più tipi senza che per questo debba esserne appositamente definito un altro; ad esempio, ogni istanza di persona può essere sia uomo sia studente
- Se si usa la generalizzazione multipla, ci si deve assicurare di rendere chiare le combinazioni “legali”
- Per questo fatto, UML pone ogni relazione di generalizzazione in un *insieme di generalizzazione*
- Sul diagramma delle classi, la freccia che indica una generalizzazione va etichettata con il nome del rispettivo insieme
- La generalizzazione singola corrisponde all'uso di un singolo anonimoinsieme di generalizzazione
- Gli insiemi di generalizzazione sono disgiunti per definizione
	- Ogni istanza del supertipo può essere istanza di uno dei sottotipi all'interno di quel sottoinsieme
 #cfigure("images/image_1710419146192_0.png", 80%)
- Questa figura ci dice che ci sono tre modi diversi di vedere una persona:
	- dal punto di vista del sesso
	- dal punto di vista dell'essere paziente o meno
	- dal punto di vista del ruolo
  
==== Relazione tra classi: sintassi

- Attenzione all'uso corretto delle frecce: UML è un linguaggio (anche se grafico) e scambiare una freccia per un'altra è un errore non da poco
  #cfigure("images/image_1710419179129_0.png", 80%)
=== Classi Astratte
- Una *#text(blue)[classe astratta]* è una classe che non può essere direttamente istanziata: per farlo bisogna prima crearne una sottoclasse concreta
- Tipicamente, una classe astratta ha una o più operazioni astratte
- Un'operazione astratta non ha implementazione
	- è costituita dalla sola dichiarazione, resa pubblica affinché le classi client possano usufruirne
- Il modo più diffuso di indicare una classe o un'operazione astratta in UML è #text(blue)[scriverne il nome in _corsivo_]
- Si possono anche rendere astratte le proprietà indicandole direttamente come tali o rendendo astratti i metodi d'accesso
- A cosa servono?
	- servono come superclassi comuni per un insieme di sottoclassi concrete
	- queste sottoclassi, in virtù del subtyping, sono in qualche misura compatibili e intercambiabili fra di loro
	- infatti sono tutte sostituibili con la superclasse
		- per il principio di sostituzione di Liskov, tutte le istanze delle sottoclassi sono anche istanze delle superclasse, quindi possono attuare qualsiasi comportamento descritto dalla superclasse, inclusi i comportamenti astratti
=== Enumerazioni
- Le enumerazioni sono usate per mostrare un insieme di valori prefissati (quelli scritti all'interno) che non hanno altre proprietà oltre al loro valore simbolico
- Sono rappresentate con una classe marcata dalla parola chiave «enumeration»
  #cfigure("images/image_1710419317953_0.png", 23%)

== Diagramma di sequenza
- Diagramma che illustra le interazioni tra le classi / entità disponendole lungo una sequenza temporale
- In particolare mostra i soggetti (chiamati tecnicamente *#text(blue)[Lifeline]*) che partecipano all'interazione e la sequenza dei messaggi scambiati
- In ascissa troviamo i diversi soggetti (non necessariamente in ordine di esecuzione), mentre in ordinata abbiamo la scala dei tempi sviluppata verso il basso
#cfigure("images/image_1710419751978_0.png", 100%)
- Le linee della _Lifeline_ partono da un quadrato, un'entità definita con sintassi UML. In questo caso i ":" indicano un'istanza della classe
- Tra la classe `User` e la classe `ACSystem` ci sarà sicuramente un'associazione
- La figura ci die inoltre che i due messaggi che vengono inviati dopo la ricezione di `PIN` possono venire inviati in momenti diversi
=== Esempio
  #cfigure("images/image_1710419777733_0.png", 80%)
	- La freccia con linea continua e piena indica un messaggio sincrono, il chiamante rimane in attesa della risposta del chiamato
	- La freccia con linea tratteggiata e vuota è il messaggio che il chiamato invia al chiamante
=== Lifeline
- In un diagramma di sequenza, i partecipanti solitamente sono istanze di classi UML caratterizzate da un nome
- La loro vita è rappresentata da una _#text(blue)[Lifeline]_, cioè una linea #text(blue)[tratteggiata verticale ed etichettata], in modo che sia possibile comprendere a quale componente del sistema si riferisce
- In alcuni casi il partecipante non è un'entità semplice, ma composta
	- è possibile modellare la comunicazione fra più sottosistemi, assegnando una Lifelinead ognuno di essi
- L'ordine in cui le OccurrenceSpecification (cioè l'invio e la ricezione di eventi) avvengono lungo la Lifeline #text(blue)[rappresenta esattamente l'ordine in cui tali eventi si devono verificare]
- La distanza (in termini grafici) tra due eventi non ha rilevanza dal punto di vista semantico
- Dal punto di vista notazionale, una Lifeline è rappresentata da un rettangolo che costituisce la “testa” seguito da una linea verticale che rappresenta il tempo di vita del partecipante
- È interessante notare che nella sezione della notazione, viene indicato espressamente che il “rettangolino” che viene apposto sulla Lifeline rappresenta l'attivazione di un metodo alla ricezione di un messaggio
==== Vincoli Temporali
- Per modellare sistemi real-time, o comunque qualsiasi altra tipologia di sistema in cui la temporizzazione è critica, è necessario specificare un istante in cui un messaggio deve essere inviato, oppure quanto tempo deve intercorrere fra un'interazione e un'altra
- Grazie, rispettivamente, a Time Constraint e Duration Constraint è possibile definire questo genere di vincoli
#cfigure("images/image_1710420323320_0.png", 90%)
*N.B.* Noi non utilizzeremo mai vincoli temporali
#cfigure("images/image_1710420355654_0.png", 100%)

=== Riferimento ad altri Diagrammi
- Spesso i diagrammi di sequenza possono assumereu una certa complessità
	- necessità di poter definire comportamenti più articolati come composizione di nuclei di interazione più semplici
- Oppure, se una sequenza di eventi ricorre spesso, potrebbe essere utile definirla una volta e richiamarla dove necessario
- Per questa ragione, UML permette di inserire #text(blue)[*riferimenti ad altri diagrammi*] e passare loro degli argomenti
- Ovviamente ha senso sfruttare quest'ultima opzione solo se il diagramma accetta dei parametri sui quali calibrare l'evoluzione del sistema
- Questi riferimenti prendono il nome di #text(blue)[InteractionUse]
- I punti di connessione tra i due diagrammi prendono il nome di #text(blue)[Gate]
- Un Gate rappresenta un #text(blue)[punto di interconnessione] che mette in relazione un messaggio al di fuori del frammento di interazione con uno all'interno del frammento
#cfigure("images/image_1710420483529_0.png", 100%)


==== Messaggio
- Un messaggio rappresenta un'interazione realizzata come comunicazione fra Lifeline
- Questa interazione può consistere nella creazione o distruzione di un'istanza, nell'invocazione di un'operazione, o nella emissione di un segnale
- UML permette di rappresentare tipi differenti di messaggi
====== Tipi di Messaggio
- Se sono specificati mittente e destinatario è un *#text(blue)[complete message]*
	- la semantica è rappresentata quindi dall'occorrenza della coppia di eventi `<sendEvent, receiveEvent>`
- Se il destinatario non è stato specificato è un *#text(blue)[lost message]*
	- in questo caso è noto solo l'evento di invio del messaggio
- Se il mittente non è stato specificato  è un *#text(blue)[found message]*
	- in questo caso è noto solo l'evento di ricezione del messaggio
- Nel caso non sia noto né il destinatario né il mittente è un *#text(blue)[unknown message]*
#cfigure("images/image_1710420606015_0.png", 90%)
- Attenzione alle frecce che usate nei messaggi, hanno significati diversi:
	- #highlight(fill: myblue)[riga continua freccia piena:] indica un messaggio (*call*) *sincrono* in cui il mittente *aspetta* il completamento dell'esecuzione del destinatario prima di continuare la sua esecuzione
	  #cfigure("images/image_1710420774542_0.png", 25%)
		- Necessita di un messaggio di ritorno per sbloccare l'esecuzione del mittente
	- #highlight(fill: myblue)[riga continua freccia vuota:] indica un messaggio *asincrono* in cui il mittente *non aspetta* il completamento dell'esecuzione del destinatario ma continua la sua esecuzione
	  #cfigure("images/image_1710420774542_1_1710421153612_0.png", 27%)
		- Il valore di ritorno potrebbe o meno essere necessario, dipende dalla semantica
	- #highlight(fill: myblue)[riga tratteggiata freccia vuota:] indica il ritorno di un messaggio
	  #cfigure("images/image_1710421186142_0.png", 27%)
	
#cfigure("images/image_1710421256766_0.png", 100%)
- Non specifichiamo chi è che ha inviato il messaggio, in questo diagramma non ci interessa
- Register manda il messaggio `doA` all'oggetto `Sale` e aspetta la sua terminazione
- Poi manda il messaggio `doB` all'oggetto `Sale` e aspetta la sua terminazione
- Infine invia il messaggio `doC`, e come conseguenza `Sale` invia `doD` alla classe `Register`
#cfigure("images/image_1710421277921_0.png", 100%)
- `authorize`
	- Per realizzare `makePayament(cashTandered)` la classe `Sale` delega la classe `Payment` (creando un istanza della classe stessa), che invoca il metodo `create(casheTandered)`; dopodiché invia il messaggio sincrono `authorize`, e restituisce il controllo a `Sale`
	- A questo punto `Sale` può notificare l'avvenuto pagamento
- `destroy`
	- con il messaggio `destroy` vine distrutta l'istanza di quella classe	
	- siccome siamo in _modellazione_, non dobbiamo soffermarci sul fatto che il nostro linguaggio di programmazione possa o meno poter distruggere un istanza di una classe


=== Combined Fragment
- La specifica di UML permette di esprimere comportamenti più complessi rispetto al singolo scambio di messaggi
- È possibile rappresentare l'esecuzione atomica di una serie di interazioni, oppure che un messaggio debba essere inviato solo in determinate condizioni
- A tale scopo UML mette a disposizione i #text(blue)[_Combined Fragment_], cioè contenitori atti a delimitare un'area d'interesse nel diagramma
- Servono per spiegare che una certa catena di eventi, racchiusa in uno o più operandi, si verificherà in base alla semantica dell'operatore associato
- Ogni fragment ha un operatore e una (eventuale) guardia

==== Tipi
- #highlight(fill: myblue)[Loop:] specifica che quello che è racchiuso nell'operando sarà eseguito ciclicamente finché la guardia sarà verificata
- #highlight(fill: myblue)[Alternatives] (alt): indica che sarà eseguito il contenuto di uno solo degli operandi, quello la cui guardia risulta verificata
- #highlight(fill: myblue)[Optional] (opt): indica che l'esecuzione del contenuto dell'operando sarà eseguita solo se la guardia è verificata
- #highlight(fill: myblue)[Break] (break): ha la stessa semantica di opt, con la differenza che in seguito l'interazione sarà terminata
- #highlight(fill: myblue)[Critical]: specifica un blocco di esecuzione atomico (non interrompibile)
- #highlight(fill: myblue)[Parallel] (par): specifica che il contenuto del primo operando può essere eseguito in parallelo a quello del secondo
- #highlight(fill: myblue)[Weak Sequencing] (seq): specifica che il risultato complessivo può essere una qualsiasi combinazione delle interazioni contenute negli operandi, purché:
	- l'ordinamento stabilito in ciascun operando sia mantenuto nel complesso
	- eventi che riguardano gli stessi destinatari devono rispettare anche l'ordine degli operandi, cioè i messaggi del primo operando hanno precedenza su quelli del secondo
	- eventi che riguardano destinatari differenti non hanno vincoli di precedenza vicendevole
- #highlight(fill: myblue)[StrictSequencing] (strict): indica che il contenuto deve essere eseguito nell'ordine in cui è specificato, anche rispetto agli operandi
- #highlight(fill: myblue)[Ignore:] indica che alcuni messaggi, importanti ai fini del funzionamento del sistema, non sono stati rappresentati, perché non utili ai fini della comprensione dell'interazione
- #highlight(fill: myblue)[Consider:] è complementare ad ignore
- #highlight(fill: myblue)[Negative] (neg): racchiude una sequenza di eventi che non deve mai verificarsi
- #highlight(fill: myblue)[Assertion] (assert): racchiude quella che è considerata l'unica sequenza di eventi valida
	- di solito è associata all'utilizzo di uno State Invariant come rinforzo
#cfigure("images/image_1710421496294_0.png", 100%)
#cfigure("images/image_1710421508172_0.png", 100%)
Fintanto che ci sono degli `item`, io faccio `enter item` e quello mi restituisce `description total`. Quando non esistono più `item` invio il messaggio `endSale`.
#cfigure("images/image_1710421519605_0.png", 90%)
Fintanto che il chiamante invoca certi numeri, ridireziona quei numeri al chiamato, se viene chiamato il 991 quella è una chiamata critica che non può essere interrotta.

== Diagrammi di Stato
- I #text(blue)[*diagrammi di stato*] modellano la dipendenza che esiste tra lo stato di una classe / entità (cioè il valore corrente delle sue proprietà) e i messaggi e/o eventi che questo riceve in ingresso
- Specifica il ciclo di vita di una classe / entità, definendo le regole che lo governano
- Quando una classe / entità si trova in un certo stato può essere interessata a determinati eventi (e non ad altri)
- Come risultato di un evento una classe / entità può passare a un nuovo stato (transizione)
- Uno #text(blue)[stato] è una condizione o situazione nella vita di un oggetto in cui esso soddisfa una condizione, esegue un'attività o aspetta un evento
- Un #text(blue)[evento] è la specifica di un'occorrenza che ha una collocazione nel tempo e nello spazio
- Una #text(blue)[transizione] è il passaggio da uno stato a un altro in risposta ad un evento
==== Esempio
#cfigure("images/image_1710421992238_0.png", 100%)
- Quando la porta è creata, la porta è aperta.
- Fintanto che la porta è aperta, è interessata a un unico messaggio: `close`; tutti gli altri messaggi vengono ignorati
- Quando riceve il messaggio `close`, e, parentesi quadra che indica la #underline[guardia], `doorway -> isEmpty()`, cioè non c'è nessuno sulla soglia, la porta transita dallo stato `open` allo stato `close`
- Dunque la porta si potrebbe trovare in un stato close, ed è interessata a due messaggi: 
	- il messaggio open, che la riporta nello stato `open`
	- il messaggio lock, che la porta nello stato `lock`
- Quando si trova nello stato lock, la porta può essere interessata solo all'evento `unlock`, che la riporta nello stato `closed`

==== Concetti
- I concetti più importanti di un diagramma di stato sono:
	- gli #highlight(fill: myblue)[stati], indicati con rettangoli con angoli arrotondati
	- le #highlight(fill: myblue)[transizioni] tra stati, indicate attraverso frecce
	- gli #highlight(fill: myblue)[eventi] che causano transizioni, la tipologia più comune è rappresentata dalla ricezione di un messaggio, che si indica semplicemente scrivendo il nome del messaggio con relativi argomenti vicino alla freccia; quindi l'oggetto, l'istanza, riceve un messaggio, e ricevendo quel messaggio effetto la transizione
	- i #highlight(fill: myblue)[marker] di #underline[inizio] e #underline[fine] rappresentati rispettivamente da un cerchio nero con una freccia che punto allo stato iniziale e come un cerchio nero racchiuso da un anello sottile
	- le #highlight(fill: myblue)[azioni] che una entità è in grado di eseguire in risposta alla ricezione di un evento; cioè, se ricevo quel evento, oltre a fare la transizione, posso fare anche altre cose
	- il #highlight(fill: myblue)[vertice] che rappresenta l'astrazione di nodo nel diagramma e può essere la sorgente o la destinazione di una o più transizioni; rappresento un sotto-diagramma
==== Esempio
#cfigure("images/image_1710422071247_0.png", 100%)
- Marker iniziale, stato iniziale, a seguito di un messaggio ricevuto, effetto quest'azione e vado in transizione verso un nuovo stato
- Da quello stato, quando posso, quando ricevo la distruzione, vado al marker finale
==== Stato
- Uno stato modella una situazione durante la quale vale una condizione invariante (solitamente implicita)
- L'invariante può rappresentare una situazione statica, come un oggetto in attesa che si verifichi un evento esterno
- Tuttavia, può anche modellare condizioni dinamiche, come il processo di esecuzione di un comportamento (cioè, l'elemento del modello in esame entra nello stato quando il comportamento inizia e lo lascia non appena il comportamento è completato)
- Stati possibili:
	- #text(blue)[Simple state]
	- #text(blue)[Composite state]
	- #text(blue)[Submachine state]
- Composite state:
	- può contenere una regione oppure è decomposto in due o più regioni ortogonali
	- ogni regione ha il suo insieme di sotto-vertici mutuamente esclusivi e disgiunti e il proprio insieme di transizioni
	- ogni stato appartenente ad una regione è chiamato substato
	- la transizione verso il marker finale all'interno di una regione rappresenta il completamento del comportamento di quella regione
	- quando tutte le regioni ortogonali hanno completato il loro comportamento questo rappresenta il completamento del comportamento dell'intero stato e fa scattare il trigger dell'evento di completamento
- Simple state:
	- è uno stato che non ha sottostati
- Submachine state:
	- specifica l'inserimento di un diagramma di una sotto-parte del sistema e permette la fattorizzazione di comportamenti comuni e il riuso dei medesimi
	- è semanticamente equivalente a un composite state, quindi le regioni del submachinestate sono come le regioni del composite state
	- le azioni sono definite come parte dello stato
- Uno stato può essere ridefinito:
	- uno stato semplice può essere ridefinito diventando uno stato composito
	- uno stato composito può essere ridefinito aggiungendo regioni, vertici, stati, transizioni e azioni


==== Esempio
#cfigure("images/image_1710422303917_0.png", 96%)
Questo è uno stato composito, nel senso che quello stato composito può essere inserito all'interno di un altro diagramma di stato. E quindi rappresenta la fattorizzazione che abbiamo visto, è analoga a quello che abbiamo visto nel diagramma dell'attività.


==== Esempio
#cfigure("images/image_1710422325220_0.png", 96%)
- Una Regione è una parte ortogonale di uno stato composto o di un state machine
- Questo è uno stato composito con regioni ortogonali che rappresenta il diagramma di stato per il superamento di un esame da parte dello studente
- In questo stato di piano ci sono le tre regioni che sono eseguite "concorrentemente", possono essere eseguite in paralleli, non sto indicando una sequenzialità
- Possono iniziare contemporaneamente ma non finiscono per forza nello stesso istante
- Il diagramma dice che il design prevede lo svolgimento della Lab1, dal quale si esce quando `lab done`
- A quel punto posso fare Lab2, quando ho fatto Lab2, `lab done`, ho terminato quella regione
- Quindi si termina quella regione quando ho terminato Lab1 e Lab2
- Nella seconda regione si esce quando il progetto è terminato
- In fine c'è il final test; dal final test si può uscire con due messaggi
	- `fail` che provoca automaticamente la segnalazione dell'esame come fallito, `Failed`
	- `pass`, allora termino quella regione, e l'esame viene segnato come passato, `Passed`



==== Esempio
#cfigure("images/image_1710422403624_0.png", 100%)


==== Pseudostato
- Gli pseudostati vengono generalmente utilizzati per collegare più transizioni in percorsi di transizioni di stato più complessi
- Ad esempio, combinando una transizione che entra in uno pseudostato fork con un insieme di transizioni che escono dallo pseudostato fork, otteniamo una transizione composta che porta a un insieme di stati ortogonali
==== Tipi di Pseudostati
- #highlight(fill: myblue)[Initial:] vertice di default che è la sorgente della singola transizione verso lo stato di default di un composite state - Ci può essere al massimo un vertice initial e la transizione uscente da questo vertice non può avere trigger o guardie
- #highlight(fill: myblue)[deepHistory]: la più recente configurazione attiva del composite state che contiene direttamente questo pseudostato
- #highlight(fill: myblue)[shallowHistory:] il più recente substate attivo di un composite state initial deepHistory shallowHistory
  #cfigure("images/image_1710422587280_0.png", 65%)
- #highlight(fill: myblue)[Join:] permette di eseguire il merge di diverse transizioni provenienti da diverse sorgenti appartenenti a differenti regioni ortogonali
	- Le transizioni entranti in questo vertice non possono avere guardie e trigger
- #highlight(fill: myblue)[Fork:] permette di separare una transizione entrante in due o più transizioni che terminano in vertici di regioni ortogonali
		- Le transizioni uscenti da questo vertice non possono avere guardie
#cfigure("images/image_1710426556871_0.png", 75%)
- #highlight(fill: myblue)[Junction:] è un vertice privo di semantica che viene usato per “incatenare” insieme transizioni multiple
	- È usato per costruire percorsi di transizione composti tra stati
	- Una junction può essere usata per far convergere transizioni multiple entranti in una singola transizione uscente che rappresenta un percorso condiviso di transizione
	- Allo stesso modo una junction può essere usata anche per suddividere una transizione entrante in più transizioni uscenti con differenti guardie
	- Permette di realizzare un #underline[branch condizionale statico], , quindi quando esco da un certo stato, so già in quale delle biforcazioni finirò
	- Esiste una guardia predefinita chiamata “else”che viene attivata quando tutte le guardie sono false
#cfigure("images/image_1710426578024_0.png", 70%)
- #highlight(fill: myblue)[Choice:] è un tipo di vertice che quando viene raggiunto causa la valutazione dinamica delle guardie dei trigger delle transizioni uscenti
	- Le guardie sono quindi tipicamente scritte sotto forma di “funzione”che viene valutata al momento del raggiungimento del vertice choice
	- Permette di realizzare un #underline[branch condizionale dinamico] separando le transizioni uscenti e creando diversi percorsi di uscita
	- Se più di una guardia è verificata viene scelto il percorso in modo arbitrario
	- Se nessuna è verificata il modello viene considerato “ill-formed”
	- È quindi caldamente consigliato definire sempre un percorso di uscita di tipo “else”
#cfigure("images/image_1710426645434_0.png", 70%)
- La differenza tra _Junciton_ e _Choise_ è che, siccome nelle transizioni io posso eseguire delle azioni, la biforcazione di junction non può dipendere dalle azioni, mentre la biforcazione di choice può dipendere dall'esito di quelle azioni.

- #highlight(fill: myblue)[Entry point:] è l'ingresso di uno state machine o di un composite state
- #highlight(fill: myblue)[Exit point:] è l'uscita da uno state machine o da uno stato composito
	- Entrare in questo vertice significa attivare la transizione che ha questo vertice come sorgente
- #highlight(fill: myblue)[Terminate:] entrare in questo vertice implica che l'esecuzione di questo state machine è terminata
	- Lo state machine non uscirà da nessuno stato né verranno invocate altre azioni a parte quelle associate con la transizione che porta allo stato terminate
#cfigure("images/image_1710426724993_0.png", 75%)
==== Transizione
- Una transizione è una relazione diretta tra un vertice di origine e un vertice di destinazione, un arco direzionato
- Può essere parte di una transizione composta, che porta la macchina a stati da una configurazione di stato a un'altra, rappresentando la risposta completa della macchina a stati al verificarsi di un evento di un tipo particolare; cioè, c'è un evento, definisco qual è la risposta della macchina a quell'evento, la risposta dell'istanza a quell'evento
- Analizzando la specifica UML è possibile vedere che tra le proprietà di una transizione compaiono diversi concetti molto rilevanti tra cui:
	- #highlight(fill: myblue)[trigger:] cioè i tipi di evento che possono innescare la transizione
	- #highlight(fill: myblue)[guardia:] cioè un vincolo che fornisce un controllo fine sull'innesco della transizione
		- La guardia è valutata quando una occorrenza dell'evento è consegnata allo stato
		- Se la guardia risulta verificata allora la transizione può essere abilitata altrimenti questa viene disabilitata
		- Il problema delle guardie è che, in teoria, la guardia potrebbe essere rappresentata dalla esecuzione di operazione; se la guardia (l'operazione) ha degli effetti collaterali (_side effect_), potrebbe succedere il caso in cui la prima ricezione di un messaggio non blocca la transizione, mentre la seconda ricezione dello stesso messaggio blocca la transizione
		- Quindi le guardie dovrebbero essere pure espressioni senza _side effect_, i quali sono assolutamente sconsigliati nella specifica
	- #highlight(fill: myblue)[effetto:] specifica un comportamento opzionale (cioè un'azione) che deve essere eseguito quando la transizione scatta trigger guardia effetto (l'effetto viene considerato da choise e non da junction)
#cfigure("images/image_1710426803459_0.png", 86%)
- Sono in stato Selling, ricevo `bid`, scatta la guardia; se il valore è minore di 100 eseguo l'operazione `reject`, e rimango nello stato selling
- Se il valore è maggiore o uguale a 100 e minore di 200 alla ricezione del messaggio `bid`, eseguo l'operazione `sell` e passo in un stato Unhappy
- Se il valore è maggiore o uguale a 200 all'atto della ricezione del messaggio `bid`, eseguo l'operazione `sell` e passo nello stato Happy



==== Tipi di Eventi
- #highlight(fill: myblue)[Evento di chiamata:] ricezione di un messaggio che richiede l'esecuzione di una operazione
- #highlight(fill: myblue)[Evento di cambiamento:] si verifica quando una condizione passa da “falsa” a “vera”
	- Tale evento si specifica scrivendo \<\<when\>\> seguito da un'espressione, racchiusa tra parentesi tonde, che descrive la condizione
	- Utile per descrivere la situazione in cui un oggetto cambia stato perché il valore dei suoi attributi è modificato dalla risposta a un messaggio inviato
- Evento segnale: consiste nella ricezione di un segnale
- Evento temporale: espressione che denota un lasso di tempo che deve trascorrere dopo un evento dotato di nome
	- Con <<after>> si può far riferimento all'istante in cui l'oggetto è entrato nello stato corrente
	- Con <<at>> si esprime qualcosa che deve accadere in un particolare momento


==== Azione
- Un'azione è un elemento avente un nome che è l'unità fondamentale della funzionalità eseguibile
- L'esecuzione di un'azione rappresenta una trasformazione o elaborazione nel sistema modellato, sia esso un sistema informatico o altro
- #highlight(fill: myblue)[Entry:] tutte le volte che si entra in uno stato viene generato un evento di entrata a cui può essere associato uno o più specifici comportamenti che vengono eseguiti prima che qualsiasi altra azione possa essere eseguita
- #highlight(fill: myblue)[Exit:] tutte le volte che si esce da uno stato viene generato un evento di uscita a cui può essere associato uno o più specifici comportamenti che vengono eseguiti come ultimo passo prima che lo stato venga lasciato
- #highlight(fill: myblue)[Do:] rappresenta il comportamento che viene eseguito all'interno dello stato
	- il comportamento parte subito dopo l'ingresso nello stato (dopo che è stato eseguito l'entry)
	- se questo comportamento termina mentre lo stato è ancora attivo viene generato un evento di completamento e nel caso sia stata specificata una transizione per il completamento allora viene eseguito l'exit e successivamente la transizione d'uscita
	- se invece viene innescata una transizione di uscita prima che il do termini, allora l'esecuzione del do è abortita, viene eseguito l'exit e poi la transizione
- Il comportamento di entry è stato definito per #text(blue)[permettere di fattorizzare comportamenti comuni] all'ingresso di uno stato, sui quali non possiamo mettere la guardia
- Senza la possibilità di poter specificare un comportamento di entry, il progettista avrebbe dovuto specificare su ogni transizione verso lo stato una azione effetto legata alle transizioni
- Il medesimo discorso vale per le exit, nella quale sono fattorizzati tutti quei comportamenti che devono essere eseguiti prima di uscire dallo stato
- È importante però ricordare che non vi è la possibilità di esprimere condizioni di guardia su questi comportamenti
==== Diagramma di Stato
- Transizione interna che esclude azioni di entrata e di uscita
  #cfigure("images/image_1710427558944_0.png", 25%)
- Attività interna: generazione di un thread concorrente che resta in esecuzione finché:
	- il thread (eventualmente) termina
	- si esce dallo stato attraverso una transizione esterna
#cfigure("images/image_1710427643942_0.png", 25%)

- Il diagramma di stato relativo a una singola classe / entità dovrebbe essere il più semplice possibile
- Le classi / entità con diagrammi di stato complicati hanno diversi problemi:
	- il codice risulta più difficile da scrivere perché le implementazioni dei metodi contengono molti blocchi condizionali
	- il testing è più arduo
	- è molto difficile utilizzare una classe dall'esterno se il comportamento dipende dallo stato in modo complesso
- Se una classe ha un numero troppo elevato di stati è bene considerare se esistano progetti alternativi, come per esempio scomporre la classe in due diverse classi
- Questa però non va presa come regola universale
- La scomposizione di una classe è sempre un'operazione molto delicata e va ponderata molto bene in quanto potrebbe portare a progetti fuorvianti o instabili
- A volte il male minore è proprio avere diagrammi di stato complessi

== Diagramma delle attività
- I diagrammi delle attività descrivono il modo  in cui diverse attività sono coordinate e possono essere  usati per mostrare l'implementazione di una operazione
- Un diagramma delle attività mostra le attività  di un sistema in generale e delle sotto-parti,  specialmente quando un sistema ha diversi obiettivi  e si desidera modellare le dipendenze tra essi  prima di decidere l'ordine in cui svolgere le azioni
- I diagrammi delle attività sono utili anche  per descrivere lo svolgimento dei singoli casi d'uso  e la loro eventuale dipendenza da altri casi
- Sostanzialmente, modellano un processo
- Organizzano più entità in un insieme di azioni  secondo un determinato flusso
- Si usano ad esempio per:
	- modellare il flusso di un caso d'uso (analisi)
	- modellare il funzionamento di un'operazione (progettazione)
	- modellare un algoritmo (progettazione)
#cfigure("images/image_1710427924392_0.png", 100%)
- #highlight(fill: myblue)[Attività:] indica un lavoro che deve essere svolto
	- Specifica un'unità atomica, non interrompibile e istantanea  di comportamento
	- Da ogni attività possono uscire uno o più archi,  che indicano il percorso da una attività ad un'altra
- #highlight(fill: myblue)[Arco:] è rappresentato come una freccia  proprio come una transizione
	- A differenza di una transizione,  un arco non può essere etichettato con eventi o azioni
	- Un arco può essere etichettato con una condizione di guardia,  se l'attività successiva la richiede
- #highlight(fill: myblue)[Oggetto:] rappresenta un oggetto “importante”  usato come input/output di azioni
- #highlight(fill: myblue)[Sottoattività:] “nasconde” un diagramma delle attività  interno a un'attività
- #highlight(fill: myblue)[Start e End Point:] punti di inizio e fine del diagramma
	- Gli End Point possono anche non essere presenti,  oppure essere più di uno
- #highlight(fill: myblue)[Controllo:] nodi che descrivono il flusso delle attività
#v(2em)
- Per capire la semantica dei diagrammi di attività,  bisogna immaginare delle entità, dette #text(blue)[token], che viaggiano lungo il diagramma
- Il flusso dei tokendefinisce il flusso dell'attività
- I token possono rimanere fermi in un nodo azione/oggetto  in attesa che si avveri
	- una condizione su un arco
	- oppure una precondizione o postcondizionesu un nodo
- Il movimento di un token è atomico
- Un nodo azione viene eseguito quando sono presenti token  su tutti gli archi in entrata, e tutte le precondizioni sono soddisfatte
- Al termine di un'azione, sono generati token su tutti gli archi in uscita
==== Notazione
#cfigure("images/image_1710428025939_0.png", 100%)
- #highlight(fill: myblue)[Decision e Merge]: determinano il comportamento condizionale
	- #highlight(fill: myblue)[Decision] ha una singola transizione entrante  e più transizioni uscenti in cui solo una di queste sarà prescelta
	- #highlight(fill: myblue)[Merge] ha più transizioni entranti e una sola uscente e serve  a terminare il blocco condizionale cominciato con un decision
- #highlight(fill: myblue)[Fork e join:] determinano il comportamento parallelo:
	- quando scatta la transazione entrante, si eseguono in parallelo  tutte le transazioni che escono dal fork
		- Con il parallelismo non è specificata la sequenza
		- Le fork possono avere delle guardie
	- Per la sincronizzazione delle attività è presente il costrutto di join che ha più transazioni entranti e una sola transazione uscente
#cfigure("images/image_1710428137459_0.png", 40%){:height 343, :width 434}


=== Object Flow
- È un arco che ha oggetti o dati che fluiscono su di esso
#cfigure("images/image_1710428157989_0.png", 100%)


=== Segnali ed Eventi
#cfigure("images/image_1710428183146_0.png", 100%)


=== Exception Handler
#cfigure("images/image_1710428204930_0.png", 100%)

#pagebreak()
=== InterruptibleActivityRegion
#cfigure("images/image_1710428254604_0.png", 100%)





= Progetto

== Analisi dei requisiti
=== Requisiti
- I requisiti di un sistema rappresentano la descrizione
	- dei servizi forniti (le funzionalità)
	- dei vincoli operativi
- Il processo di ricerca, analisi, documentazione e verifica di questi servizi e vincoli è chiamato *Ingegneria dei Requisiti* (RE - Requirements Engineering)
- I requisiti di solito vengono forniti a livelli diversi di descrizione e questo porta a una prima classificazione...
- I Requisiti utente dichiarano:
	- Quali servizi il sistema dovrebbe fornire
	- I vincoli sotto cui deve operare
	- Sono requisiti molto astratti e di alto livello che vengono specificati nella prima fase interlocutoria con il committente
	- Tipicamente sono espressi in linguaggio naturale e corredati da qualche diagramma
- Requisiti di sistema definiscono:
	- Le funzioni, i servizi e i vincoli operativi del sistema in modo dettagliato
	- È una descrizione dettagliata di quello che il sistema dovrebbe fare
	- Il Documento dei Requisiti del Sistema deve essere preciso e definire esattamente cosa deve essere sviluppato
	- Tale documento fa spesso parte del contratto tra committente e azienda sviluppatrice
- ![image.png](images/image_1710063362882_0.png)
==== Requisiti di Sistema
- I requisiti di sistema tipicamente vengono divisi in tre diverse tipologie:
	- Requisiti funzionali
	- Requisiti non funzionali
	- Requisiti di dominio
==== Requisiti Funzionali
- Descrivono quello che il sistema dovrebbe fare
- Sono elenchi di servizi che il sistema dovrebbe fornire
- Per ogni servizio dovrebbe essere indicato:
	- come reagire a particolari input
	- come comportarsi in particolari situazioni
	- in alcuni casi specificare cosa il sistema NON dovrebbe fare
- Le specifiche dei requisiti funzionali dovrebbero essere:
	- complete: tutti i servizi definiti
	- coerenti: i requisiti non devono avere definizioni contraddittorie
==== Requisiti Non Funzionali
- Non riguardano direttamente le specifiche funzionali
- Non devono essere necessariamente coerenti
- I principali tipi di requisiti non funzionali:
	- Requisiti del prodotto: specificano o limitano le proprietà complessive del sistema
	- affidabilità, prestazioni, protezione dei dati, disponibilità dei servizi, tempi di risposta, occupazione di spazio, capacità dei dispositivi di I/O, rappresentazione dei dati nelle interfacce, etc.
	- Requisiti Organizzativi: possono vincolare anche il processo di sviluppo adottato
	- politiche e procedure dell'organizzazione cliente e sviluppatrice, specifiche degli standard di qualità da adottare, uso di un particolare CASE toole linguaggi di implementazione, limiti di budget, requisiti di consegna e milestone, etc.
	- Requisiti esterni: si identificano tutti i requisiti che derivano da fattori non provenienti dal sistema e dal suo processo di sviluppo
		- necessità di interoperabilità con altri sistemi software o hardware prodotti da altre organizzazioni
		- requisiti legislativi che devono essere rispettati affinché il sistema operi nella legalità →legislazioni sulla privacy dei dati
		- requisiti etici perché il sistema possa essere accettato dagli utenti e dal grande pubblico
- ![image.png](images/image_1710063695471_0.png)
- Uno dei maggiori problemi di questi requisiti è che possono essere difficili da verificare perché spesso sono espressi in modo vago e sono mescolati ai requisiti funzionali
	- Esempi: facilità d'uso, capacità di ripristino dopo un malfunzionamento
- Spesso contrastano o interagiscono con i requisiti funzionali
	- La protezione e la privacy dei dati contrastano con la facilità d'uso perché richiedono procedure spesso macchinose per l'utente...
	- ...occorre trovare un compromesso tra i requisiti o chiedere al committente quale sia più prioritario
- Vanno studiati ed analizzati con molta cura e precisione e indicati quanto più chiaramente possibile nel documento dei requisiti
==== Requisiti di Dominio
- I requisiti di dominio descrivono i vincoli che esistono nel dominio e che potrebbero non essere condivisi tra tutti coloro che saranno impegnati nello sviluppo del software
- Derivano dal dominio di applicazione del sistema
- Solitamente includono una terminologia propria del dominio del sistema o si riferiscono ai suoi concetti
- Poiché sono requisiti “specialistici” spesso gli ingeneri del software trovano difficile capire come questo tipo di requisiti si rapportino agli altri requisiti del sistema
- Sono requisiti che vanno comunque analizzati con molta cura perché riflettono i fondamenti del dominio dell'applicazione
	- l'analisi deve coinvolgere gli esperti del dominio per chiarire ogni dubbio sulla terminologia


=== Analisi

- Obiettivo
	- Specificare (cioè definire) le proprietà che il sistema dovrà avere senza descrivere una loro possibile realizzazione
- Risultato: una serie di documenti
	- contenenti la descrizione dettagliata dei requisiti
	- base di partenza per lanalisi del problema
- Per determinare in dettaglio i requisiti del sistema, è necessario
	- interagire il più possibile con lutente, o con il potenziale utente
	- conoscere il più possibile larea applicativa
+ #highlight(fill: myblue)[Raccolta dei requisiti] \
  Obiettivo: raccogliere tutte le informazioni su cosa il sistema deve fare secondo le intenzioni del cliente
+ #highlight(fill: myblue)[Analisi dei requisiti] \
  Obiettivo: definire il comportamento del sistema
+ #highlight(fill: myblue)[Analisi del dominio] \
  Obiettivo: definire la porzione del mondo reale, rilevante per il sistema
+ #highlight(fill: myblue)[Analisi e gestione dei rischi] \
  Obiettivo: identificare e gestire i possibili rischi che possono fare fallire o intralciare la realizzazione del sistema
 ![image.png](images/image_1710064987414_0.png)
==== Raccolta dei Requisiti
- Obiettivo:
  raccogliere tutte le informazioni su cosa il sistema deve fare secondo le intenzioni del cliente (si chiede al cliente cosa voglia che faccia l'applicazione)
- Non prevede passi formali, né ha una notazione specifica, perché dipende moltissimo dal particolare tipo di problema
- Risultato
	- un documento (testuale)
		- scritto dallanalista
		- discusso e approvato dal cliente
	- una versione iniziale del vocabolario o glossario contenente la definizione precisa e non ambigua di tutti i termini e i concetti utilizzati
- Tipologie di persone coinvolte
	- Analista
	- Utente
	- Esperto del dominio (non sempre indispensabile)
- Metodi utilizzati
	- Interviste, questionari
	- Studio di documenti che esprimono i requisiti in forma testuale
	- Osservazione passiva o attiva del processo da modellare
	- Studio di sistemi software esistenti
	- Prototipi
- La gestione delle interviste è molto complessa
- I clienti potrebbero
	- Avere solo una vaga idea dei requisiti
	- Non essere in grado di esprimere i requisiti in termini comprensibili
	- Chiedere requisiti non realizzabili o troppo costosi
	- Fornire requisiti in conflitto
	- Essere poco disponibili a collaborare
==== Validazione dei Requisiti
- Ogni requisito deve essere validato e negoziato con i clienti prima di essere riportato nel Documento dei Requisiti (se non è completamente validato si rifà l'intervista)
- Attività svolta in parallelo alla raccolta
- #highlight(fill: myblue)[Validità] - il nuovo requisito è inerente il problema da risolvere?
- #highlight(fill: myblue)[Consistenza] - il nuovo requisito è in sovrapposizione e/o in conflitto con altri requisiti?
- #highlight(fill: myblue)[Realizzabilità] - il nuovo requisito è realizzabile con le risorse disponibili (hardware, finanziamenti, ...)?
- #highlight(fill: myblue)[Completezza] - esiste la possibilità che ci siano requisiti rimasti ignorati?
==== Documento dei Requisiti
- Il Documento dei Requisiti deve specificare in modo chiaro e univoco cosa il sistema dovrà fare
- I requisiti dovrebbero essere:
	- Chiari
	- Corretti
	- Completi
	- Concisi
	- Non ambigui
	- Precisi
	- Consistenti
	- Tracciabili
	- Modificabili
	- Verificabili
- Il Documento dei Requisiti deve contenere la versione iniziale del dizionario dei termini
- Un buon modo per organizzare i requisiti e facilitare la tracciabilità è quello di elencare i requisiti in una tabella
- ![image.png](images/image_1710065295316_0.png)
- Se durante il processo di sviluppo ci si riferisce sempre allid del requisito diventa facile collegare le diverse fasi e garantire una tracciabilità requisito-codice
==== Cambiamento dei Requisiti
- È normale che i requisiti subiscano modificazioni ed evolvano nel tempo
- Requisiti esistenti possono essere rimossi o modificati
- Nuovi requisiti possono essere aggiunti in una fase qualsiasi del ciclo di sviluppo
- Tali cambiamenti
	- Sono la norma, non l'eccezione; i sistemi vanno progettati avendo in mente il cambiamento
	- Possono diventare un grosso problema se non opportunamente gestiti
- Più lo sviluppo è avanzato, più il cambiamento è costoso
	- Modificare un requisito appena definito è facile
	- Modificare lo stesso requisito dopo che è stato implementato nel software potrebbe essere molto costoso
- Ogni cambiamento deve essere accuratamente analizzato per valutare
	- La fattibilità tecnica, cioè, lo si può implementare, indipendentemente dal costo?
	- Limpatto sul resto del sistema
	- Il costo
- Consiglio - sviluppare sistemi che
	- Siano il più possibile resistenti ai cambiamenti dei requisiti
		- Inizialmente, eseguano esclusivamente e nel modo migliore i soli compiti richiesti
		- In seguito, siano in grado di sostenere laggiunta di nuove funzionalità senza causare “disastri” strutturali e/o comportamentali
- Tenete traccia dei cambiamenti anche nella tabella dei requisiti
==== Analisi del Dominio
- Obiettivo:
  definire la porzione del mondo reale rilevante per il sistema, quindi una visione astratta del mondo reale
- Principio fondamentale: Astrazione
  Permette di gestire la complessità intrinseca
  del mondo reale
	- ignorare gli aspetti che non sono importanti per lo scopo attuale
	- concentrarsi maggiormente su quelli che lo sono
- Risultato:
  prima versione del vocabolario partendo dai “sostantivi” che si trovano nei requisiti
- L'analisi del dominio può essere effettuata anche considerando un gruppo di sistemi afferenti alla stessa area applicativa
- Esempi di aree applicative:
	- il controllo del traffico aereo
	- la gestione aziendale
	- le operazioni bancarie
	- ...
- In tal caso, è possibile
	- identificare entità e comportamenti comuni a tutti i sistemi
	- realizzare schemi di progettazione e componenti software riutilizzabili nei diversi sistemi
==== Analisi dei Requisiti
- Obiettivo:
  definire il comportamento del sistema da realizzare
- Risultato:
  un modello comportamentale (o modello dinamico) che descrive in modo chiaro e conciso le funzionalità del sistema
- L'analisi dei requisiti deve partire dalla raccolta, che è un elenco di requisiti, e definire per bene qual è il comportamento del sistema
- Strategia:
	- Scomposizione funzionale (mediante analisi top-down) ► identificare le singole funzionalità previste dal sistema
	- Astrazione procedurale ► considerare ogni operazione come una singola entità, nonostante tale operazione sia effettivamente realizzata da un insieme di operazioni di più basso livello
- Attenzione:
  La scomposizione in funzioni è molto volatile a causa del continuo cambiamento dei requisiti funzionali
- Come prima cosa vanno analizzati in modo sistematico tutti i requisiti inseriti nella Tabella dei requisiti
- Bisogna porre particolare attenzione ai sostantivi e ai verbi che compaiono nel testo di specifica dei requisiti
- Dall'analisi dei sostantivi è possibile formalizzare la conoscenza sul dominio applicativo → costruzione di un primo modello del dominio
- Dall'analisi dei verbi è possibili individuare linsieme delle azioni che il sistema dovrà compiere → modello dei casi duso
- Aggiornare costantemente la Tabella dei Requisiti → dallanalisi nascono sempre nuovi requisiti
==== Vocabolario
- Nella modellazione del dominio è di fondamentale importanza usare solo la terminologia di quello specifico dominio
- Il vocabolario è una lista dei termini usati nella specifica dei requisiti a cui viene data una definizione precisa e non ambigua
- È uno dei fattori chiave per migliorare la comunicazione tra i diversi attori del processo di sviluppo, in particolare tra analisti e progettisti
- Ciascuna entità del dominio che si evince dai requisiti può essere espressa come classe UML e messa in relazione logica con le altre entità andando a creare il primo modello del dominio
==== Analisi e gestione dei rischi
- Ci si riferisce generalmente ai rischi di tipo legale
- Analisi sistematica e completa di tutti i possibili rischi che possono fare fallire o intralciare la realizzazione del sistema in una qualsiasi fase del processo di sviluppo
- Ogni rischio presenta due caratteristiche:
	- Probabilità che avvenga
	  non esistono rischi con una probabilità del 100% (sarebbero vincoli al progetto)
	- Costo
	  se il rischio si realizza, ne seguono effetti indesiderati e/o perdite
- Rischi relativi ai requisiti
	- I requisiti sono perfettamente noti?
	- Il rischio maggiore è quello di costruire un sistema che non soddisfa le esigenze del cliente
- Rischi relativi alle risorse umane
	- È possibile contare sulle persone e sullesperienza necessarie per lo sviluppo del progetto?
- Rischi relativi alla protezione e privacy dei dati
	- Di che tipo sono i dati che dobbiamo trattare?
	- Quali sono i possibili attacchi informatici a cui il sistema può essere soggetto?
- Rischi tecnologici
	- Si sta scegliendo la tecnologia corretta?
	- Si è in grado di aggregare correttamente i vari componenti del progetto (ad es., GUI, DB, ...)?
	- Quali saranno i possibili futuri sviluppi della tecnologia?
- Rischi politici
	- Ci sono delle forze politiche (anche in senso lato) in grado di intralciare lo sviluppo del progetto?
- Strategia reattiva o “alla Indiana Jones”
	- “Niente paura, troverò una soluzione”
- Strategia preventiva
	- Si mette in moto molto prima che inizi il lavoro tecnico
	- Si individuano i rischi potenziali, se ne valutano le probabilità e gli effetti e si stabilisce un ordine di importanza
	- Si predispone un piano che permetta di reagire in modo controllato ed efficace
		- Più grande è un rischio, dove per grande possiamo intendere il prodotto di probabilità per costo
		- Maggiore sarà l'attenzione che bisognerà dedicargli

=== Sicurezza e Privacy
==== Nuova normativa
===== General Data Protection Regulation

- Dal 25/5/2018 sostituisce la Data Protection Directive
- Obbligo di aderenza (compliance) di un prodotto software che tratti dati personali ai principi della GDPR
	- privacy by design & by default(misure tecniche e organizzative)
	- minimalità, proporzionalità
	- anonimizzazione, pseudonimizzazione
	- trasferimento dati fuori dalla EU (occhio al cloud!)
	- adeguatezza delle misure di sicurezza
- Non è qualcosa che si possa “aggiungere dopo”, a sistema già progettato: va considerato fin dall'inizio
	- ma non è (solo) un vincolo: è un'opportunità per creare valore

#highlight(fill: myblue)[Pseudonimizzazione:]
processo di trattamento dei dati personali
in modo tale che i dati non possano più
essere attribuiti a un interessato specifico
senza l'utilizzo di informazioni aggiuntive,
sempre che tali informazioni aggiuntive
siano conservate separatamente
e soggette a misure tecniche e organizzative
intese a garantire la non attribuzione
a una persona identificata o identificabile.\


===== Principi
I dati personali devono essere:

- trattati in modo lecito, equo e trasparente nei confronti dell'interessato (“#text(blue)[liceità, equità e trasparenza]”)

- raccolti per finalità determinate, esplicite e legittime, e successivamente trattati in modo non incompatibile con tali finalità

- adeguati, pertinenti e limitati a quanto necessario rispetto alle finalità per le quali sono trattati (“#text(blue)[minimizzazione dei dati]”)

- esatti e, se necessario, aggiornati
	- devono essere prese tutte le misure ragionevoli per cancellare o rettificare tempestivamente i dati inesatti rispetto alle finalità per le quali sono trattati (“#text(blue)[esattezza]”); quindi bisogna dare all'utente la possibilità di aggiornarli

- conservati in una forma che consenta l'identificazione degli interessati per un arco di tempo non superiore al conseguimento delle finalità per le quali sono trattati
	- i dati personali possono essere conservati per periodi più lunghi a condizione che siano trattati esclusivamente per finalità di archiviazione nel pubblico interesse o per finalità di ricerca scientifica e storica o per finalità statistiche, conformemente all'articolo 83

- trattati in modo da garantire un'adeguata sicurezza dei dati personali, compresa la protezione, mediante misure tecniche e organizzative adeguate, da trattamenti non autorizzati o illeciti e dalla perdita, dalla distruzione o dal danno accidentali (“#text(blue)[integrità e riservatezza]”); quindi bisogna garantire che non ci siano trattamenti illeciti dei dati degli utenti, e che non vengano persi

===== Articolo 25
*Protezione dei dati fin dalla progettazione e protezione di default*

+ Tenuto conto dello stato dell'arte e dei costi di attuazione, nonché della natura, del campo di applicazione, del contesto e delle finalità del trattamento, come anche dei rischi di varia probabilità e gravità per i diritti e le libertà delle persone fisiche costituiti dal trattamento, #text(blue)[sia al momento di determinare i mezzi del trattamento sia all'atto del trattamento stesso] il responsabile del trattamento mette in atto #text(blue)[misure tecniche e organizzative adeguate], quali la pseudonimizzazione, volte ad attuare i principi di protezione dei dati, quali la minimizzazione, in modo efficace e a integrare nel trattamento le necessarie garanzie al fine di soddisfare i requisiti del presente regolamento e tutelare i diritti degli interessati
+ Il responsabile del trattamento mette in atto misure tecniche e organizzative adeguate per garantire che siano trattati, #text(blue)[di default], solo i dati personali necessari per ogni specifica finalità del trattamento; ciò vale per la quantità dei dati raccolti, l'estensione del trattamento, il periodo di conservazione e l'accessibilità. #text(blue)[In particolare dette misure garantiscono che, di default, non siano resi accessibili dati personali a un numero indefinito di persone fisiche senza l'intervento della persona fisica] ; cioè, normalmente, nessuno può far vedere quei dati, tranne la persona stessa

===== Articolo 32:
*Sicurezza del Trattamento*

+ Tenuto conto dello stato dell'arte e dei costi di attuazione, nonché della natura, del campo di applicazione, del contesto e delle finalità del trattamento, come anche del rischio di varia probabilità e gravità per i diritti e le libertà delle persone fisiche, il responsabile del trattamento e l'incaricato del trattamento mettono in atto misure tecniche e organizzative adeguate per garantire un #text(blue)[livello di sicurezza adeguato al rischio], che comprendono tra l'altro:
	- la #text(blue)[pseudonimizzazione] e la #text(blue)[cifratura dei dati personali];
	- la capacità di assicurare la #text(blue)[continua riservatezza], integrità, disponibilità e resilienza dei sistemi e dei servizi che trattano i dati personali;
	- la capacità di #text(blue)[ripristinare tempestivamente la disponibilità] e l'accesso dei dati in caso di incidente fisico o tecnico;
	- una procedura per provare, verificare e valutare regolarmente l'efficacia delle misure tecniche e organizzative al fine di garantire la sicurezza del trattamento; cioè bisogna non solo mettere in campo queste misure, ma anche testarle regolarmente
+ Nel valutare l'adeguato livello di sicurezza, si tiene conto in special modo [dei rischi presentati] da trattamenti di dati derivanti in particolare dalla distruzione, dalla perdita, dalla modifica, dalla divulgazione non autorizzata o dall'accesso, in modo accidentale o illegale, a dati personali trasmessi, memorizzati o comunque trattati

+ L'adesione a un #text(blue)[codice di condotta] approvato di cui all'articolo 40 o a un #text(blue)[meccanismo di certificazione] approvato di cui all'articolo 42 può essere utilizzata come elemento per dimostrare la conformità ai requisiti di cui al paragrafo 1 del presente articolo

Quindi per dimostrare di aver seguito tutti i passi descritti nell'articolo ci si avvale di una certificazione o di un codice di condotta.

===== Trasferimento Dati a Paesi Terzi

- Articolo 45:
	-  Il trasferimento di dati personali verso un paese terzo o un'organizzazione internazionale è ammesso se la Commissione ha deciso che il paese terzo, o un territorio o uno o più settori specifici all'interno del paese terzo, o l'organizzazione internazionale in questione garantiscano un livello di protezione adeguato. In tal caso il trasferimento non necessita di autorizzazioni specifiche.


- Articolo 46:
	-  In mancanza di una decisione ai sensi dell'articolo 45, il responsabile del trattamento o l'incaricato del trattamento può trasferire dati personali verso un paese terzo o un'organizzazione internazionale solo se ha offerto garanzie adeguate e a condizione che siano disponibili diritti azionabili degli interessati e mezzi di ricorso effettivi per gli interessati.

==== Concetti di Base
===== Sicurezza Informatica
Salvaguardia dei sistemi informatici da potenziali rischi e/o violazioni dei dati

- Obiettivo dell'attacco = contenuto informativo
- Sicurezza informatica = preoccuparsi di
	-  impedire l'accesso ad utenti non autorizzati
	-  regolamentare l'accesso ai diversi soggetti...
	-  ... che potrebbero avere autorizzazioni diverse (relative solo a certe operazioni e non altre)
- per evitare che i dati appartenenti al sistema informatico vengano copiati, modificati o cancellati


Cosa Proteggere:

- L'informazione...
	-  Riservatezza: solo determinati utenti possono avere accesso alle informazioni
	-  Integrità ed Autenticità: nessuno deve modificare impropriamente quelle informazioni
	-  Disponibilità: le informazioni sono disponibili nel momento c'è la necessità di accedervi 
- trattata per mezzo di calcolatori e reti
	-  Accesso controllato
		- Tre aspetti: #highlight(fill: myblue)[identificazione, autenticazione, autorizzazione]

	- Funzionamento affidabile; per il concetto di affidabilità riguardare gli argomenti precedenti


===== Violazioni

- Le violazioni possono essere molteplici:
	-  tentativi non autorizzati di accesso a zone riservate
	-  furto di identità digitale o di file riservati
	-  utilizzo di risorse che l'utente non dovrebbe potere utilizzare
	-  ecc.


- La sicurezza informatica si occupa anche di prevenire eventuali #text(blue)[Denial of Service] (DoS)
	- sono attacchi sferrati al sistema con l'obiettivo di rendere inutilizzabili alcune risorse onde danneggiare gli utenti

===== Fattori Influenti
- Nella scelta delle misure di sicurezza #text(red)[incidono diverse caratteristiche] dell'informazione e del contesto:
	- Dinamicità
	- Dimensione e tipo di accesso
	- Tempo di vita
	- Costo di generazione
	- Costo in caso di violazione (di riservatezza, di integrità, di disponibilità)
	- Valore percepito (dall'utente e dal detentore) e tipologia di attaccante

#heading(level: 5, numbering: none)[La Catena degli Anelli]
La forza di un sistema è pari alla forza dell'anello più debole che lo compone. Non bisogna tralasciare nulla, nemmeno il più piccolo dettaglio

===== Metodi di Protezione

- Legali: ci sono leggi che regolano i casi in cui un sistema informatico viene violato, attraverso sanzioni ad esempio
- Organizzativi
- Tecnologici: metodi che ci interessano di più
	-  Fisici: un oggetto fisico che viene utilizzato per proteggere l'accesso ai dati, ad esempio un tesserino
	-  Crittografici: cifrare i dati in modo che siano difficilmente interpretabili da un attaccante
	-  Biometrici: caratteristiche fisiche misurabili che identificano univocamente un individuo

Attuiamo questi metodi per:
- Prevenzione: evitare che un attaccante entri nel sistema
- Rilevazione: scopriamo eventuali attacchi o violazioni
- Contenimento: se c'è una violazione dei sistemi si cerca di tutti i modi di contenere eventuali danni
- Ripristino: ripristino del sistema all'istante precedente al danno


===== Protezione Fisica
- Essenziale: la vulnerabilità fisica non sia la più facilmente attaccabile

- Efficace per i sistemi
	- criteri che esistono da ben prima dei problemi di sicurezza informatica
- Efficace per i dati
	-  purché si conosca il comportamento dei sistemi che li trattano (percorsi accessibili, copie temporanee in memoria e su disco, ...)
	-  costo di generazione
	-  purché si prevedano metodi aggiuntivi per contenere gli effetti delle violazioni fisiche dei sistemi (es. furto)

===== Autenticazione Forte

#cfigure("images/2024-03-18-10-26-42.png", 90%)
L'autenticazione forte è garantita esistono le seguenti condizioni
- Possesso
- Conoscenza
- Conformità

Ovviamente garantire tutte e tre le condizioni ha un costo.

==== Crittografia

- Crittografia simmetrica
	-  garantire riservatezza
	-  non identifica né autentica
	-  *funzionamento*: esiste una sola chiave che viene utilizzata per cifrare; ad esempio prendo un dato, lo cifro, e ottengo un dato che non è immediatamente comprensibile; la stessa chiave è usata per decifrare.
- Crittografia asimmetrica
	-  garantisce riservatezza
	-  obiettivo: identificazione e quindi autenticazione e paternità
	-  *funzionamento*: le chiavi sono due; c'è una che è in esclusivo possesso del cifratore e una che invece è pubblica, quindi chiave privata e chiave pubblica
- Infrastrutture per la certificazione della chiave pubblica
	-  Terze parti fidate che possano certificare l'autenticità di una chiave pubblica

===== Crittografia Simmetrica
#cfigure("images/2024-03-18-10-52-29.png", 90%)
- Classica e moderna, implementata con dispositivi segreti, algoritmi segreti o chiavi segrete
- Tipicamente tecniche derivanti dalla teoria dell'informazione (confusione e diffusione)
- Una singola chiave cifra e decifra

===== Crittografia Asimmetrica

- Moderna (ufficialmente 1976)
- Basata sulla teoria della complessità computazionale
- Due chiavi correlate ma non (facilmente) calcolabili l'una dall'altra
	- La #text(red)[*chiave privata*] è strettamente personale e quindi _identifica_ il possessore
	- L'uso di una determinata chiave privata può essere verificato da chiunque per mezzo della corrispondente #text(green)[*chiave pubblica*]

#figure(image("images/2024-03-18-10-58-12.png", width: 90%), caption: "Procedimento di cifratura")
Se io voglio che il mio documento sia letto solamente da chi deve poterlo leggere, io cifro quel documento con la chiave pubblica, quindi se io cifro con la chiave privata è decifrabile solo dalla chiave pubblica, se io cifro con la chiave pubblica è decifrabile solo dalla chiave privata


*Firma Digitale:*

#figure(image("images/2024-03-18-11-01-47.png", width: 90%), caption: "Procedimento di firma")

Viceversa, firma, autenticità: se io cifro il documento con la mia chiave pubblica genero un documento che è leggibile, ma che di fianco ha la mia firma, a questo punto chiunque sia in possesso della chiave pubblica può cifrare o decifrare la firma, e vedere se i documenti sono uguali, cioè controllo che la chiave che è stata usata per cifrare, per produrre la firma, corrisponde alla chiave pubblica che ho in mano io.


===== Confronto tra i due tipo di crittografia

#cfigure("images/2024-03-18-11-07-55.png", 85%)
La cifratura asimmetrica purtroppo è poco efficiente perché le procedure di cifratura e decifratura evidentemente sono complicate. Però, per quanto riguarda le chiavi simmetriche, bisogna averne un numero importante, perché se ho una sola chiave che cifra e decifra tutto, se capita in mano a un malintenzionato ha accesso a tutto.

===== Biometria

- Componente cardine per la terna di fattori per l'#text(blue)[*autenticazione forte*]
	- qualcosa che sei, qualcosa che hai, qualcosa che sai
- Problemi non ancora risolti:
	- Ostinatamente usata per l'_identificazione_
		- lunghe operazioni di confronto
		- cattivo bilanciamento tra falsi positivi e falsi negativi
	- Meglio per l'_autenticazione_
		- un solo confronto, minore probabilità di errori
	- Prestazioni più sfumate rispetto ad altre tecniche
		- difficile tuning tra falsi positivi e falsi negativi
	- Impossibilità di sostituzione in caso di compromissione

===== Sicurezza delle Password

- Aspetto da sempre fondamentale per:
	-  impedire l'accesso a utenti non autorizzati
	-  nascondere e/o vincolare l'accesso a documenti
- Diverse categorie:
	-  Deboli
	-  Semplici
	-  Intelligenti
	-  Strong

==== Analizzare e Progettare la Sicurezza

===== Sistema Sicuro

- La definizione di una politica di sicurezza deve tenere conto di vincoli tecnici, logistici, amministrativi, politici ed economici, imposti dalla struttura organizzativa in cui il sistema opera

- Per questo serve introdurre la sicurezza sin dalle prime fasi di analisi dei requisiti di un nuovo sistema

	- le vigenti leggi, le politiche e i vincoli aziendali sono la base di partenza per la definizione di un #text(blue)[*piano per la sicurezza*]

- Un'applicazione o un servizio possono consistere di uno o più componenti funzionali allocati localmente o distribuiti sulla rete

- La sicurezza viene vista come un #text(blue)[processo complesso], come una catena di caratteristiche
	- dalla computer system security, network security, application-level security sino alle problematiche di protezione dei dati sensibili
- La sfida maggiore lanciata ai progettisti è quella di #text(blue)[progettare applicazioni sicure e di qualità] che tengano conto in modo strutturato di tutti gli aspetti della sicurezza sin dalle prime fasi di analisi del sistema


===== Sistemi Critici

- I #text(blue)[sistemi critici] sono sistemi tecnici o socio-tecnici da cui dipendono persone o aziende

- Se questi sistemi non forniscono i loro servizi come ci si aspetta possono verificarsi seri problemi e importanti perdite

- Ci sono tre tipi principali di sistemi critici:
	-  #highlight(fill: myblue)[Sistemi safety-critical:] i fallimenti possono provocare incidenti, perdita di vite umane o seri danni ambientali
	-  #highlight(fill: myblue)[Sistemi mission-critical:] i malfunzionamenti possono causare il fallimento di alcune attività e obiettivi diretti
	-  #highlight(fill: myblue)[Sistemi business-critical:] i fallimenti possono portare a costi molto alti per le aziende che li usano

- La proprietà più importante di un sistema critico è la sua #text(blue)[*fidatezza*]

- Sistema fidato = #text(red)[*disponibilità + affidabilità + sicurezza e protezione*]

- Ci sono diverse ragioni per le quali la fidatezza è importante:
	- I sistemi non affidabili, non sicuri e non protetti sono rifiutati dagli utenti
	- I costi di un fallimento del sistema potrebbero essere enormi
	- I sistemi inaffidabili possono causare perdita di informazioni


- Componenti del sistema che possono causare fallimenti (in ordine dal più affidabile al meno affidabile):
	-  Hardware: può fallire a causa di errori nella progettazione, di un guasto a un componente o perché i componenti hanno terminato la loro vita naturale
	-  Software: può fallire a causa di errori nelle sue specifiche, nella sua progettazione o nella sua implementazione
	-  Operatori umani: possono sbagliare a interagire con il sistema


- Con l'aumentare dell'affidabilità di software e hardware gli errori umani sono diventati la più probabile causa di difetto di un sistema. Una persona può non usarlo come dovrebbe, come abbiamo pensato che dovesse utilizzarlo o che volontariamente usa il sistema in maniera inopportuna per evitare la fidatezza del sistema e quindi per distruggere l'affidabilità, distruggere la sicurezza, distruggere la protezione e distruggere la disponibilità.

- La sicurezza e la protezione dei sistemi critici sono diventate sempre più importanti con l'aumentare delle connessioni di rete

- Da una parte le connessioni di rete espongono il sistema ad attacchi da parte di malintenzionati

- Dall'altra parte la rete fa in modo che i dettagli delle vulnerabilità siano facilmente divulgati e facilita la distribuzione di patch

- Esempi di attacchi sono:
	-  virus
	-  usi non autorizzati dei servizi
	-  modifiche non autorizzate al sistema e ai suoi dati
	- #highlight(fill: myblue)[Exploit:] metodo che sfrutta un bug o una vulnerabilità, per l'acquisizione di privilegi
	- #highlight(fill: myblue)[Buffer Overflow:] fornire al programma più dati di quanto esso si aspetti di ricevere, in modo che una parte di questi vadano scritti in zone di memoria dove sono, o dovrebbero essere, altri dati o lo stack del programma stesso
	- #highlight(fill: myblue)[Shell code:] sequenza di caratteri che rappresenta un codice binario in grado di lanciare una shell, può essere utilizzato per acquisire un accesso alla linea di comando
	- #highlight(fill: myblue)[Sniffing:] attività di intercettazione passiva dei dati che transitano in una rete
	- #highlight(fill: myblue)[Cracking:] modifica di un software per rimuovere la protezione dalla copia, oppure per ottenere accesso a un'area riservata
	- #highlight(fill: myblue)[Spoofing:] tecnica con la quale si simula un indirizzo IP privato da una rete pubblica facendo credere agli host che l'IP della macchina server da contattare sia il suo
	- #highlight(fill: myblue)[Trojan:] programma che contiene funzionalità maliziose; la vittima è indotta a eseguire il programma poiché questo viene spesso inserito nei videogiochi pirati
	- #highlight(fill: myblue)[Denial of Service:] il sistema viene forzatamente messo in uno stato in cui i suoi servizi non sono disponibili, influenzando così la disponibilità del sistema

==== Introduzione alla Security Engineering

===== Security Engineering

#cfigure("images/2024-03-18-11-53-45.png", 90%)

- L'ingegneria della sicurezza è parte del più vasto campo della sicurezza informatica
- Nell'ingegnerizzazione di un sistema software #underline[non si può] prescindere dalla consapevolezza delle #underline[minacce] che il sistema dovrà affrontare e dei modi in cui tali minacce possono essere neutralizzate


- Quando si considerano le problematiche di sicurezza nell'ingegnerizzazione di un sistema vanno presi in considerazione due aspetti diversi:
	- la sicurezza dell'applicazione
	- la sicurezza dell'infrastruttura su cui il sistema è costruito

#cfigure("images/2024-03-18-12-16-43.png", 90%)

===== Applicazione e Infrastruttura

- La sicurezza di una applicazione è un problema di ingegnerizzazione del software dove gli ingegneri devono garantire che il #text(blue)[*sistema sia progettato per resistere agli attacchi*]
- La sicurezza dell'infrastruttura è invece un problema manageriale nel quale gli amministratori dovrebbero garantire che #text(blue)[*l'infrastruttura sia configurata per resistere agli attacchi*]

- Gli amministratori dei sistemi devono:
	-  inizializzare l'infrastruttura in modo tale che tutti i servizi di sicurezza siano disponibili
-  monitorare e riparare eventuali falle di sicurezza che emergono durante l'uso del software

===== Gestione della Sicurezza

- Gestione degli utenti e dei permessi:
	-  inserimento e rimozione di utenti dal sistema
	-  autenticazione degli utenti
	-  creazione di appropriati permessi per gli utenti
- Deployment e mantenimento del sistema:
	-  installazione e configurazione dei software e del middleware
	-  aggiornamento periodico del software con tutte le patch disponibili
- Controllo degli attacchi, rilevazione e ripristino
	-  controllo del sistema per accessi non autorizzati
	-  identificazione e messa in opera di strategie contro gli attacchi
	-  backup per ripristinare il normale utilizzo dopo un attacco

==== Glossario e Minacce
===== Glossario della Sicurezza

- #highlight(fill: myblue)[Bene] (Asset): una risorsa del sistema che deve essere protetta
- #highlight(fill: myblue)[Esposizione] (Exposure): possibile perdita o danneggiamento come risultato di un attacco andato a buon fine
	- Potrebbe essere una perdita o un danneggiamento di dati o una perdita di tempo nel ripristino del sistema dopo l'attacco
- #highlight(fill: myblue)[Vulnerabilità] (Vulnerability): una debolezza nel sistema software che potrebbe essere sfruttata per causare una perdita o un danno

- #highlight(fill: myblue)[Attacco] (Attack): sfruttamento di una vulnerabilità del sistema allo scopo di esporre un bene
- #highlight(fill: myblue)[Minaccia] (Threat): circostanza che ha le potenzialità per causare perdite e danni

- #highlight(fill: myblue)[Controllo] (Control): una misura protettiva che riduce una vulnerabilità del sistema

===== Tipi di Minacce

- #text(blue)[Minacce alla riservatezza del sistema o dei suoi dati]: le informazioni posso essere svelate a persone o programmi non autorizzati

- #text(blue)[Minacce all'integrità del sistema o dei suoi dati]: i dati o il software possono essere danneggiati o corrotti

- #text(blue)[Minacce alla disponibilità del sistema o dei suoi dati]: può essere negato l'accesso agli utenti autorizzati al software o ai dati

- Queste minacce sono interdipendenti
	-  Se un attacco rende il sistema non disponibile, la modifica sulle informazioni potrebbe non avvenire, rendendo così il sistema non integro

===== Tipi di Controllo

- #text(blue)[Controlli per garantire che gli attacchi non abbiano successo]: la strategia è quella di progettare il sistema in modo da evitare i problemi di sicurezza
	- i sistemi militari sensibili non sono connessi alla rete pubblica
	- crittografia
- #text(blue)[Controlli per identificare e respingere attacchi]: la strategia è quella di monitorare le operazioni del sistema e identificare pattern di attività atipici, nel caso agire di conseguenza (spegnere parti del sistema, restringere l'accesso agli utenti, ..)

- #text(blue)[Controlli per il ripristino]
	-  backup, replicazione, polizze assicurative

#heading(level: 5, numbering: none)[Esempio]

- Un sistema informativo ospedaliero mantiene le informazioni personali sui pazienti e sui loro trattamenti

- Il sistema deve essere accessibile da differenti ospedali e cliniche attraverso un'interfaccia web

- Lo staff ospedaliero deve utilizzare una specifica coppia \<username, password\> per autenticarsi, dove lo username è il nome del dipendente

- Il sistema richiede password che siano lunghe almeno 8 caratteri, ma consente ogni password senza ulteriori verifiche

#figure(image("images/2024-03-18-12-28-51.png"), caption: "Schema logico")

#cfigure("images/2024-03-18-12-29-51.png", 100%)

==== Analisi del Rischio

- L'analisi del rischio si occupa di 
	- valutare le possibili perdite che un attacco può causare ai beni di un sistema 
	- bilanciare queste perdite con i costi richiesti per la protezione dei beni stessi

#cfigure("images/2024-03-18-12-31-20.png", 90%)

- L'analisi del rischio è una problematica più manageriale che tecnica

- Il ruolo degli ingegneri della sicurezza è quindi quello di fornire una #text(blue)[guida tecnica e giuridica] sui problemi di sicurezza del sistema

- Sarà poi compito dei manager decidere se accettare i costi della sicurezza o i rischi che derivano dalla mancanza di procedure di sicurezza

- L'analisi del rischio inizia dalla valutazione delle politiche di sicurezza di organizzazione che spiegano cosa dovrebbe e cosa non dovrebbe essere consentito fare

- Le politiche di sicurezza propongono le condizioni che dovrebbero sempre essere mantenute dal sistema di sicurezza, quindi aiutano ad identificare le minacce che potrebbero sorgere

- La valutazione del rischio è un processo in più fasi:
	-  #text(blue)[valutazione preliminare del rischio]: determina i requisiti di sicurezza dell'intero sistema
-  #text(blue)[ciclo di vita della valutazione del rischio]: avviene contestualmente e segue il ciclo di vita dello sviluppo del software; valutare i rischi ogni volta che viene cambiato qualcosa

#figure(image("images/2024-03-18-12-34-44.png"), caption: "Valutazione Preliminare del Rischio")


==== Identificazione del bene
===== Analisi del Sistema Informatico
- Durante questa fase si può stabilire la seguente agenda delle attività:
	- Analisi delle risorse fisiche
	- Analisi delle risorse logiche
	- Analisi delle dipendenze fra risorse

===== Analisi delle Risorse Fisiche

- In questa attività, il sistema informatico viene visto come insiemi di dispositivi che, per funzionare, hanno bisogno di spazio, alimentazione, condizioni ambientali adeguate, protezioni da furti e danni materiali

- In particolare occorre:
	-  individuare sistematicamente tutte le risorse fisiche
	-  ispezionare e valutare tutti i locali che ospiteranno le risorse fisiche
	-  verificare la cablatura dei locali

===== Analisi delle Risorse Logiche

- Il sistema viene visto come insieme di informazioni, flussi e processi

- In particolare occorre:
	-  #text(red)[Classificare le informazioni] in base al valore che hanno per l'organizzazione, il grado di riservatezza e il contesto di afferenza
	-  #text(red)[Classificare i servizi] offerti dal sistema informatico affinché non presentino effetti collaterali pericolosi per la sicurezza del sistema

===== Analisi delle Dipendenze tra Risorse

- Per ciascuna risorsa (fisica o logica) occorre individuare di quali altre risorse essa ha bisogno per funzionare correttamente

- Questa analisi tende ad evidenziare le risorse potenzialmente critiche, ovvero quelle da cui dipende il funzionamento di un elevato numero di altre risorse

- I risultati di questa analisi sono usati anche nella fase di valutazione del rischio
	- in particolare, sono di supporto allo studio della propagazione dei malfunzionamenti a seguito dell'occorrenza di eventi indesiderati

==== Identificazione delle minacce

- In questa fase si cerca di definire quello che #text(red)[#underline[non deve poter accadere]] nel sistema

- Si parte dal considerare come evento indesiderato qualsiasi accesso che non sia esplicitamente permesso

- A tal fine è possibile in generale distinguere tra
	-  attacchi intenzionali
	-  eventi accidentali

===== Attacchi Intenzionali

- Gli attacchi vengono caratterizzati in funzione della risorsa (sia fisica che logica) che viene attaccata e delle possibili tecniche usate per l'attacco

- Le tecniche di attacco possono essere classificate in funzione del livello al quale operano

- Si distingue tra #text(red)[tecniche a livello fisico] e #text(red)[a livello logico]
	- Gli attacchi a livello fisico sono principalmente tesi a sottrarre o danneggiare le risorse critiche
	- Si tratta di
		-  Furto = un attacco alla disponibilità e alla riservatezza
		-  Danneggiamento = un attacco alla disponibilità e alla integrità

===== Attacchi a Livello Logico

- Gli attacchi a livello logico sono principalmente tesi a sottrarre informazione o degradare l'operatività del sistema

- Dal punto di vista dei risultati che è indirizzato a conseguire un attacco logico può essere classificato come:

- #highlight(fill: myblue)[Intercettazione e deduzione] (attacco alla riservatezza): sniffing, spoofing, emulatori...
- #highlight(fill: myblue)[Intrusione] (attacco all'integrità e alla riservatezza): IP-spoofing, backdoor...
- #highlight(fill: myblue)[Disturbo] (attacco alla disponibilità): virus, worm, denial of service...

===== Eventi Accidentali

- Una possibile casistica degli eventi accidentali che accadono più di frequente:

- #text(red)[a livello fisico]:
	-  guasti ai dispositivi che compongono il sistema
	-  guasti di dispositivi di supporto (es. condizionatori)
- #text(red)[a livello logico]:
	-  perdita di password o chiave hardware
	-  cancellazione di file
	-  corruzione del software di sistema (ad esempio, a seguito dell'installazione di estensioni incompatibili)

===== Valutazione dell'Esposizione

- A ogni minaccia occorre associare un rischio così da indirizzare l'attività di individuazione delle contromisure verso le aree più critiche

- Per #text(red)[rischio] s'intende una combinazione della probabilità che un evento accada con il danno che l'evento può arrecare al sistema

- Nel valutare il danno si tiene conto
	-  delle dipendenze tra le risorse
	-  dell'eventuale propagazione del malfunzionamento

===== Valutazione delle Probabilità: Attacchi Intenzionali

- La probabilità di occorrenza di attacchi intenzionali dipende principalmente dalla #text(red)[facilità] di attuazione e dai #text(red)[vantaggi] che potrebbe trarne l'intruso
	- Il danno si misura come #text(blue)[grado di perdita] dei tre requisiti fondamentali (riservatezza, integrità, disponibilità)

- MA l'attaccante applicherà sempre tutte le tecniche di cui dispone, su tutte le risorse attaccabili → necessità di valutare anche il rischio
di un #text(red)[attacco composto]
	- un insieme di attacchi elementari concepiti con un medesimo obiettivo e condotti in sequenza

===== Individuazione del Controllo

- Occorre scegliere il controllo da adottare per neutralizzare gli attacchi individuati:
	- Valutazione del rapporto costo/efficacia
	- Analisi di standard e modelli di riferimento
	- Controllo di carattere organizzativo
	- Controllo di carattere tecnico

===== Valutazione del Rapporto Costo/Efficacia

- Valuta il #text(red)[grado di adeguatezza] di un controllo
- Mira ad evitare che i controlli presentino un costo ingiustificato rispetto al rischio dal quale proteggono

- #text(red)[Efficacia del controllo] definita come funzione del rischio rispetto agli eventi indesiderati che neutralizza

- Il costo di un controllo deve essere calcolato senza dimenticare i #text(blue)[costi nascosti]


===== Costi Nascosti

- Occorre tenere presenti le limitazioni che i controlli impongono e le operazioni di controllo che introducono nel flusso di lavoro del sistema informatico e dell'organizzazione

- Le principali voci di costo sono le seguenti:
	-  costo di messa in opera del controllo
	-  peggioramento dell'ergonomia dell'interfaccia utente
	-  decadimento delle prestazioni del sistema nell'erogazione dei servizi
	-  aumento della burocrazia

===== Controlli di Carattere Organizzativo

- #text(blue)[Condizione essenziale] affinché la tecnologia a protezione del sistema informatico risulti efficace è che venga #text(blue)[utilizzata nel modo corretto] da personale pienamente #text(blue)[consapevole]

- Devono quindi essere definiti con precisione #text(red)[ruoli] e #text(red)[responsabilità] nella gestione sicura di tale sistema

- Per ciascun ruolo, dall'amministratore al semplice utente, devono essere definite #text(red)[norme comportamentali] e #text(red)[procedure precise] da rispettare

===== Controlli di Carattere Tecnico

- Controlli di base
	-  a livello del sistema operativo e dei servizi di rete
- Controlli specifichi del particolare sistema
	-  si attestano normalmente a livello applicativo
- Controlli tecnici più frequenti
	-  configurazione sicura del sistema operativo di server e postazioni di lavoro (contromisura di base)
	-  confinamento logico delle applicazioni server su server dedicati


- Etichettatura delle informazioni, allo scopo di avere

un controllo più fine dei diritti di accesso

- Moduli software di cifratura integrati con le applicazioni
- Apparecchiature di telecomunicazione in grado

di cifrare il traffico dati in modo trasparente
alle applicazioni

- Firewall e server proxy in corrispondenza

di eventuali collegamenti con reti TCP/IP

- Chiavi hardware e/o dispositivi di riconoscimento

degli utenti basati su rilevamenti biofisici
Integrazione dei Controlli

- Un insieme di controlli non deve presentarsi come

una “collezione di espedienti” non correlati tra loro

- È importante integrare i vari controlli

in una politica di sicurezza organica

- Operare una selezione dei controlli adottando

un sottoinsieme di costo minimo che rispetti alcuni vincoli:

- completezza delle contromisure
- omogeneità delle contromisure
- ridondanza controllata delle contromisure
- effettiva attuabilità delle contromisure

Vincoli del Sottoinsieme

- Completezza:

il sottoinsieme deve fare fronte a tutti
gli eventi indesiderati

- Omogeneità:

le contromisure devono essere compatibili
e integrabili tra loro

- Ridondanza controllata:

la ridondanza delle contromisure ha un costo
e deve essere rilevata e vagliata accuratamente

- Effettiva attuabilità:

l'insieme delle contromisure deve rispettare tutti i vincoli
imposti dall'organizzazione nella quale andrà ad operare

Esempio

- Torniamo all'esempio del sistema per la gestione

di dati ospedalieri

- Quali sono i beni del sistema?
-  il sistema informativo
-  il database dei pazienti
-  i record di ogni paziente

• Quali sono le minacce?

- furto identità dell'amministratore
- furto identità di utente autorizzato
- DoS

Esempio

Valutazione del Bene

Bene Valore Esposizione

Sistema

Informativo

Alto. Supporto a tutte le
consultazioni cliniche

Alta. Perdita finanziaria; costi
ripristino sistema; danni a
pazienti se cure non date

Database
pazienti

Alto. Supporto a tutte le
consultazioni cliniche
Critico dal punto di vista della
sicurezza

Alta. Perdita finanziaria; costi
ripristino sistema; danni a
pazienti se cure non date

Record
paziente

Normalmente basso.
Potrebbe essere alto per
pazienti particolari

Perdita diretta bassa, ma
possibile perdita di reputazione
della clinica

Esempio

Analisi Minacce e Controlli

Minaccia Probabilità Controllo Fattibilità

Furto identità

Amministratore

Bassa Accesso solo da
postazioni specifiche
fisicamente sicure

Basso costo
implementativo ma
attenzione alla
distribuzione delle chiavi

Furto identità
utente

Alta Meccanismi biometrici Molto costoso e non
accettato
Log di tutte le
operazioni

Semplice, trasparente e
supporta il ripristino

DoS Media Progettazione
adeguata, controllo e
limitazione degli accessi

Basso costo. Impossibile
prevedere e impedire
questo tipo di attacco

Esempio

Requisiti di Sicurezza

- Alcuni dei requisiti ricavati dalla valutazione preliminare

dei rischi

- le informazioni relative ai pazienti devono essere scaricate
- all'inizio della sessione clinica dal database e memorizzate
- in un'area sicura sul client
- le informazioni relative ai pazienti non devono essere mantenute
- sul client dopo la chiusura della sessione clinica
- deve essere mantenuto un log
- su una macchina diversa dal server
- che memorizzi tutti i cambiamenti effettuati sul database

Bene Valore Esposizione
Sistema Informativo Alto. Supporto a tutta la
gestione del villaggio turistico

Alta. Perdita finaziaria e di
immagine
Informazioni relative
agli Ospiti

Medio. Dati generali degli ospiti
del villaggio turistico

Media. Perdita di immagine se
vengono divulgati dati degli Ospiti

Informazioni relative
alle GuestCard

Alto. L'elenco degli acquisti è
associato alle GuestCard

Molto Alta. Perdita finanziaria nel
caso gli Ospiticontestino acquisti
ingiustamente addebitati. Perdita di
immagine
Informazioni relative
alle vendite

Alto. Sulla basedei movimenti
nei PuntidiVendita,
la CatenaPuntiVenditagestisce
i magazzini e i dati fiscali

Alta. Perdita finanziaria se i dati
delle vendite e delle forniture non
coincidono. Perdita di immagine se
il servizio che vuole essere
acquistato per qualche motivo non
è presente
Informazioni relative al
personale

Alto. Ci sono tutte le
informazioni relative al
personale, comprese le
credenziali di accesso

Alta. Perdita finanziaria, se
vengono rubate le credenziali del
personale si possono perpetuare
svariate frodi. Perdita di immagine

Villaggio Turistico

Valutazione del Bene
Villaggio Turistico

Analisi Minacce e Controlli
Minaccia Probabilità Controllo Fattibilità
Furto credenziali
Operatore

Alta. La username è fissata e
facile da identificare

Accesso da
macchine sicure

Basso costo di realizzazione ma
attenzione se le macchine vengono
lasciate incustodite
Log delle Operazioni Basso costo implementativo

Furto credenziali
personale dei
punti di vendita

Alta. La username è fissata e
facile da identificare

Accesso da
macchine sicure

Basso costo di realizzazione ma
attenzione se le macchine vengono
lasciate incustodite
Log delle Operazioni Basso costo implementativo

Intercettazione
comunicazioni

Alta. Il sistema è distribuito e
client/server avvengono
tantissime interazioni tra i
diversi client e il server.

Cifratura delle
comunicazioni

Il costo dipende dal tipo di cifratura
scelto. Se simmetrica basso, se
asimmetrica più alto dovuto alla
necessità di rilascio di coppie di
chiavi da un ente di certificazione
Log di tutte le
operazioni

Basso costo implementativo

DoS Bassa Progettazione
adeguata, controllo e
limitazione degli
accessi

Basso costo. Impossibile prevenire
un DoS

Ciclo di vita della valutazione

del rischio

Ciclo di Vita della Valutazione del Rischio

- È necessaria la conoscenza dell'architettura del sistema
- e dell'organizzazione dei dati
- La piattaforma e il middleware sono già stati scelti,
- così come la strategia di sviluppo del sistema
- Questo significa che si hanno molti più dettagli
- riguardo a che cosa è necessario proteggere
- e sulle possibili vulnerabilità del sistema
- Le vulnerabilità possono essere “ereditate”
- dalle scelte di progettazione ma non solo
- La valutazione del rischio dovrebbe essere parte
- di tutto il ciclo di vita del software: dall'ingegnerizzazione
- dei requisiti al deployment del sistema

Ciclo di Vita della Valutazione del Rischio

- Il processo seguito è simile a quello della valutazione
- preliminare dei rischi, con l'aggiunta di attività riguardanti
- l'identificazione e la valutazione delle vulnerabilità
- La valutazione delle vulnerabilità identifica i beni
- che hanno più probabilità di essere colpiti
- da tali vulnerabilità
- Vengono messe in relazione le vulnerabilità
- con i possibili attacchi al sistema
- Il risultato della valutazione del rischio è
- un insieme di decisioni ingegneristiche
- che influenzano la progettazione o l'implementazione
- del sistema o limitano il modo in cui esso è usato

Esempio

- Si supponga che il provider dei servizi ospedalieri

decida di acquistare un prodotto commerciale
per la gestione dei dati dei pazienti

- Questo sistema deve essere configurato

per ogni tipo di clinica in cui è utilizzato

- Scelte progettuali del sistema acquistato:
-  autenticazione solo con username e password
-  architettura client-server: il client accede attraverso
-  un'interfaccia web standard
-  l'informazione è presentata agli utenti
-  attraverso una web form editabile,
-  è quindi possibile modificare le informazioni

Esempio: Vulnerabilità

Autenticazione

Login/password

Tecnologia Vulnerabilità

Password
banali

Utente rivela
password

Architettura

client-server

Denial of
Service

Informazioni
nella cache

Uso web form
editabili

Log dettagliati
non possibili

Stessi permessi
per gli utenti

Bachi nel
browser

Esempio

- Valutate le vulnerabilità del sistema adottato

si deve decidere quali mosse attuare
al fine di limitare i rischi associati

- Introduzione di nuovi requisiti di sicurezza
- Introduzione di un meccanismo di verifica delle password
-  l'accesso al sistema deve essere permesso
-  solo ai client approvati e registrati dall'amministratore
-  tutti i client devono avere un solo browser installato
-  e approvato dall'amministratore

Villaggio Turistico

Vulnerabilità

Tecnologia Vulnerabilità

Autenticazione
username/password

- Password banali
- Utente rivela volontariamente password
- Utente rivela password a seguito di un attacco di
- Ingegneria Sociale

Cifratura comunicazioni Le vulnerabilità dipendono dal tipo di cifratura.
Cifratura Simmetrica:

- Tempo di vita della chiave. Più informazioni cifro
- con la stessa chiave più materiale offro per l'analisi
- del testo ad un attaccante
- Lunghezza della chiave
- Memorizzazione della chiave
- Cifratura Asimmetrica:
- Memorizzazione chiave privata

Architettura Client/Server • DoS

- Man in the Middle
- Sniffing delle comunicazioni

Security Use Case e Misuse Case
Security Use Case e Misuse Case

- I misuse case si concentrano sulle interazioni

tra l'applicazione e gli attaccanti che cercano di violarla

- La condizione di successo di un misuse case

è l'attacco andato a buon fine

- Questo li rende particolarmente adatti per analizzare

le minacce, ma non molto utili per la determinazione
dei requisiti di sicurezza

- È invece compito dei security use case specificare

i requisiti tramite i quali l'applicazione
dovrebbe essere in grado di proteggersi dalle minacce

Security Use Cases -Donald G. Firesmith-JOT Vol. 2, No. 3, May-June 2003

Security Use Case e Misuse Case

MisuseCase Security Use Case

Uso Analizzare e specificare le
minacce

Analizzare e specificare i
requisiti di sicurezza

Criteri di successo Successo attaccanti Successo applicazione

Prodotto da Security Team Security Team

Usato da Security Team RequirementsTeam

Attori Esterni Attaccanti, Utenti Utenti

Guidato da Analisi vulnerabilità dei beni e
analisi minacce

Misusecase

Security Use Case e Misuse Case
Esempio

INTRODUCTION

VOL. 2 , NO. 3 JOURNAL OF OBJECT TECHNOLOGY 55

The following table summarizes the primary differences between misuse cases and
security use cases.

Misuse Cases Security Use Cases
Usage Analyze^ and^ specify^ security^
threats.

Analyze and specify security
requirements
Success Criteria Misuser^ Succeeds^ Application^ Succeeds^
Produced By Security^ Team^ Security^ Team^
Used By Security^ Team^ Requirements^ Team^
External Actors Misuser,^ User^ User^
Driven By Asset^ Vulnerability^ Analysis^
Threat Analysis

Misuse Cases

To further illustrate the differences between normal use cases, security use cases, and
associated misuse cases, consider Figure 3. The traditional use cases for an automated
teller machine might include Deposit Funds, Withdraw Funds, Transfer Funds, and Query
Balance, all of which are specializations of a general Manage Accounts use case. To
securely manage one's accounts, one can specify security use cases to control access
(identification, authentication, and authorization), ensure privacy (of data and
communications), ensure integrity (of data and communications), and ensure
nonrepudiation of transactions. The resulting four security use cases specify requirements
that protect the ATM and its users from three security threats involving attacks by either
crackers or thiefs.

Customer
Manage
Accounts

Deposit
Funds

Withdraw
Funds

Transfer
Funds

Query
Balance

Control Access
(Security)

Ensure Privacy
(Security)

Ensure Integrity
(Security)

Ensure
Nonrepudiation
(Security)

Spoof User
(Misuse)

Invade Privacy
(Misuse)

Perpetrate Fraud
(Misuse)

Cracker

Thief

Misuse
Case Misuser

Security
Use Case

Fig. 3 : Example Security Use Cases and Misuse Cases
Security Use Cases -Donald G. Firesmith-JOT Vol. 2, No. 3, May-June 2003
Esempio: Scenario

Security Use Cases -Donald G. Firesmith-JOT Vol. 2, No. 3, May-June 2003
Villaggio Turistico

Punto Vendita

CatenaPuntiVendita

ElencoVendite

Registrazione

NuovoMovimento

Login

ChiusuraCredito

GestioneOspite

ElencoAcquisti
FineSettimana<<event>>

FineVacanza<<event>>

Operatore

OspitePagante

<<include>>

<<extend>>

<<extend>>

AperturaCredito
<<extend>>

<<include>><<extend>>

<<include>>

Gestione Villaggio Turistico

Garantire Protezione (security) Controllo Accesso (security)

(misuse)Frode SniffingInformazioni(misuse) FurtoCredenziali(misuse)

Truffatore Hacker

Villaggio Turistico

Titolo ControlloAccesso
Descrizione Gliaccessialsistemadevonoesserecontrollati
Misusecase SniffingInformazioni,FurtoCredenziali
Relazioni
Precondizioni L'AttaccantehaimezziperscoprirealmenolausernamediOperatoriepersonaledeipuntivendita

Postcondizioni IlSistemabloccamomentaneamentel'accessoall'utenteenotificauntentativodiaccessofraudolento
Scenario
principale

Sistema Attaccante
Dopo aver scoperto qualche username
tenta di accedere al sistema inserendo
passwordconunattaccocondizionario
Controllalecredenzialiimmesseebloccal'accessonelcasotali
credenzialirisultinoerratedopouncertonumeroditentativi.
Scenario di
attacco avvenuto
consuccesso

Sistema Attaccante
Attaccocondizionarioriuscito
IlSistemacontrollalecredenzialiimmesseeconsentel'accesso
alsistema
Naviga tra le maschere del sistema e
cercadicarpirepiùinformazionipossibili
IlSistemascrivenellogtutteleoperazionieseguitedall'utente
Il Sistema controllaperiodicamente il logallaricerca di pattern
diaccessoatipicieserilevatinotificaunaccessofraudolento

Security Use Case: Linee Guida

- I casi d'uso non dovrebbero mai specificare meccanismi

di sicurezza

- Le decisioni relative ai meccanismi devono essere lasciate
- alla progettazione
- Requisiti attentamente differenziati

dalle informazioni secondarie

- interazioni del sistema, azioni del sistema e post-condizioni
- sono i soli requisiti
- Evitare di specificare vincoli progettuali non necessari
- Documentare esplicitamente i percorsi individuali attraverso

i casi d'uso al fine di specificare i reali requisiti di sicurezza

- Basare i security use case su differenti tipi di requisiti di sicurezza

fornisce una naturale organizzazione dei casi d'uso
Security Use Case: Linee Guida

- Documentare le minacce alla sicurezza che giustificano

i percorsi individuali attraverso i casi d'uso

- Distinguere chiaramente tra interazioni degli utenti

e degli attaccanti

- Distinguere chiaramente tra le interazioni

che sono visibili esternamente e le azioni nascoste
del sistema

- Documentare sia le precondizioni che le post-condizioni

che catturano l'essenza dei percorsi individuali
SPECIFICA DEI REQUISITI DI

SICUREZZA

Requisiti di Sicurezza

- Non è sempre possibile specificare i requisiti

associati alla sicurezza in modo quantitativo

- Quasi sempre questa tipologia dei requisiti

è espressa nella forma “non deve”

- definisce comportamenti inaccettabili per il sistema
- non definisce funzionalità richieste al sistema
- L'approccio convenzionale della specifica dei requisiti

è basato sul contesto, sui beni da proteggere
e sul loro valore per l'organizzazione

Categorie Requisiti di Sicurezza

- Requisiti di identificazione
-  specificano se un sistema deve identificare gli utenti
-  prima di interagire con loro
- Requisiti di autenticazione
-  specificano come identificare gli utenti
- Requisiti di autorizzazione
-  specificano i privilegi e i permessi di accesso
-  degli utenti identificati
- Requisiti di immunità
-  specificano come il sistema deve proteggersi da virus, worm
-  e minacce simili

Categorie Requisiti di Sicurezza

- Requisiti di integrità
-  specificano come evitare la corruzione dei dati
- Requisiti di scoperta delle intrusioni
-  specificano quali meccanismi utilizzare per scoprire
-  gli attacchi al Sistema
- Requisiti di non-ripudiazione
-  specificano che una parte interessata in una transazione
-  non può negare il proprio coinvolgimento
- Requisiti di riservatezza
-  specificano come deve essere mantenuta la riservatezza
-  delle informazioni

Categorie Requisiti di Sicurezza

- Requisiti di controllo della protezione
-  specificano come può essere controllato e verificato
-  l'uso del sistema
- Requisiti di protezione della manutenzione del sistema
-  specificano come una applicazione può evitare
-  modifiche autorizzate da un accidentale annullamento
-  dei meccanismi di protezione

Villaggio Turistico

Requisiti di Sicurezza

- Creazione di un log per tracciare
-  tutte le azioni che avvengono sul sistema
-  i messaggi scambiati tra le parti del sistema
-  - che vanno protetti in un qualche modo per evitare che un accesso
-  - fraudolento al sistema di log possa rivelare dati riservati
- Adottare meccanismi di analisi del log per
-  identificare pattern di accesso atipici
-  identificare discrepanze tra i messaggi spediti e ricevuti
- Individuare una corretta politica di controllo degli accessi
- I dati memorizzati e scambiati nel sistema

devono essere protetti


== Analisi del problema

=== Introduzione

- #text(red)[Obiettivo]:
    esprimere fatti il più possibile “oggettivi”
    sul #text(blue)[#underline[*problema*]] focalizzando l'attenzione su #text(blue)[sottosistemi], #text(blue)[ruoli] e #text(blue)[responsabilità] insiti negli scenari
    prospettati durante l'analisi dei requisiti
    #text(blue)[#underline[senza descrivere la sua possibile soluzione]]
- #text(red)[Risultato]:
    - Architettura Logica
    - Piano di Lavoro
    - Piano del Collaudo (in che modo collauderemo il sistema)

#cfigure("images/2024-03-22-10-41-17.png", 100%)

=== Passi dell'Analisi del Problema
1. #text(blue)[Analisi del Documento dei Requisiti]\
    Obiettivo: concentrarsi sull'analisi delle funzionalità
    e dei rischi evidenziati nel documento dei requisiti
2. #text(blue)[Analisi dei Ruoli e delle Responsabilità]\
    Obiettivo: analizzare bene i ruoli emersi nei casi d'uso
    e porre particolare attenzione nell'attribuzione
    delle responsabilità a tali ruoli tenendo conto dell'analisi
    del rischio
3. #text(blue)[Scomposizione del problema]\
    Obiettivo: se possibile suddividere il problema
    in sotto-problemi più piccoli
4. #text(blue)[Creazione del Modello del Dominio]\
    Obiettivo: partendo dal vocabolario si costruisce
    il diagramma delle classi del dominio; il modello di dominio è il modello delle entità che sono importanti nel dominio applicativo e definisce le relazioni tra tali entità
5. #text(blue)[Architettura Logica: Struttura]\
    Obiettivo: creazione dei diagrammi strutturali
    (package e classi) dell'Architettura Logica
6. #text(blue)[Architettura Logica: Interazione]\
    Obiettivo: creazione dei diagrammi di interazione
    (diagrammi di sequenza) dell'Architettura Logica; specificano le funzionalità
7. #text(blue)[Architettura Logica: Comportamento]\
    Obiettivo: creazione dei diagrammi di comportamento
    (stato e attività) dell'Architettura Logica, se è opportuno che ci siano
8. #text(blue)[Definizione del Piano di Lavoro]\
    Obiettivo: assegnare le responsabilità
    a ciascun membro del team di progetto,
    stabilire i vincoli temporali,
    impostare eventuali milestone, ...
9. #text(blue)[Definizione del Piano del Collaudo]\
    Obiettivo: definire i risultati attesi da ogni entità
    (classe, sottosistema) che compare nell'Architettura
    Logica; scrivere le classi di test per verificare tali risultati

#cfigure("images/2024-03-22-10-43-41.png", 100%)

==== Architettura Logica
- L'*#text(blue)[Architettura Logica]* è un insieme di modelli UML
    costruiti per definire una struttura concettuale
    del problema robusta e modificabile
- Attraverso l'Architettura Logica l'analista descrive
    - #text(blue)[la struttura] (insieme delle parti)
    - #text(blue)[il comportamento atteso] (da tutto e da ciascuna parte)
    - #text(blue)[le interazioni]
così come possono essere dedotte dai requisiti,
dalle caratteristiche del problema e del dominio
applicativo *#text(red)[senza alcun riferimento alle tecnologie
realizzative]*


==== Piano di Lavoro

- Il #text(blue)[Piano di Lavoro] esprime l'articolazione delle attività
    in termini di risorse
       - umane
       - temporali
       - di elaborazione
       - ...
necessarie allo sviluppo del prodotto
in base ai risultati dell'Analisi del Problema


==== Piano del Collaudo

- Il #text(blue)[Piano del Collaudo] definisce l'insieme dei risultati attesi
    da ogni entità definita nell'Architettura Logica
    in relazione a specifiche sollecitazioni stimolo-risposta prevista
- Un modo per impostare il Piano del Collaudo
    è quello di fornire i test delle classi definite nell'Architettura Logica,
    così da pianificare già in questa fase i test di integrazione
    dei vari sottosistemi
- Ciò agevola il lavoro della successiva fase di progetto,
    riducendo il rischio di brutte sorprese in fase di integrazione
    delle sotto-parti
- Strumenti tipici per scrivere i test-case:
    - JUnit nel mondo Java
    - NUnit(http://nunit.org/) nel mondo C\#

#pagebreak()
=== Analisi Documento dei Requisiti

#cfigure("images/2024-03-22-10-46-37.png", 100%)

- Il Documento dei Requisiti evidenzia
    - le #text(blue)[funzionalità] e i #text(blue)[servizi] che dovranno essere sviluppati (requisiti funzionali)
    - i #text(blue)[vincoli] che dobbiamo tenere in considerazione (requisiti non funzionali)
    - il “#text(blue)[mondo esterno]” con cui si dovrà interagire (attori e sistemi esterni)
    - i “#text(blue)[rischi]” legati a possibili attacchi alla protezione, integrità e privacy dei dati (requisiti di sicurezza)
- Partendo da tale documento occorre applicare
    un'attenta analisi delle singole funzionalità, vincoli,
    interazioni e rischi

#cfigure("images/2024-03-22-10-48-27.png", 95%)

==== Analisi delle Funzionalità

- Per ogni funzionalità espressa nei Casi d'Uso vanno analizzati:
    - _#text(blue)[il tipo, ovvero l'obiettivo della funzionalità]_ : memorizzazione dei dati,
       interazione con l'esterno, gestione/manipolazione dei dati
          - etichettiamo la funzionalità in modo da rendere evidente
             durante la creazione dell'Architettura Logica dove “posizionarla”
          - se la funzionalità è molto complessa potrebbe appartenere a più tipi
             differenti:marcare la funzionalità come “complessa”
    - #text(blue)[_le informazioni coinvolte_]: analisi sistematica di tutte le informazioni
       sulle quali deve “operare” la funzionalità
          - In particolare individuare il tipo di dato (composto/singolo)
             e il grado di riservatezza richiesto
    - _#text(blue)[il “flusso delle informazioni”]_ : quali sono le informazioni che
       - #underline[si ricevono in input]: permette di dare indicazioni sulla validazione
          dell'input e se sono presenti vincoli sui valori ammessi
       - #underline[vanno prodotte in output]: permette di valutare dall'esterno
          se la funzionalità si comporta come ci si aspetta


==== Analisi delle Funzionalità

- Si possono predisporre diverse tabelle di analisi

#cfigure("images/2024-03-22-10-51-43.png", 100%)
Se il grado di complessità è alto (Funzionalità complessa), allora bisognerà scomporla in sotto-funzionalità.

- La Tabella delle Funzionalità è una sola tabella per tutte le funzionalità
- La Tabella delle Informazioni/Flusso è una tabella per ogni funzionalità; è importante specificare ogni funzionalità, anche se prima si sono tralasciate funzionalità semplici

==== Esempio

- Tabella Funzionalità per il Villaggio Turistico

#cfigure("images/2024-03-22-10-53-26.png", 100%)

- Tabella Informazioni/Flusso per NuovoMovimento

#cfigure("images/2024-03-22-10-54-04.png", 100%)

==== Analisi dei Vincoli

- Per ogni requisito non funzionale vanno analizzati:
    - #text(blue)[il tipo, ovvero a quale categoria appartiene]:
       vincoli sulle performance, sui tempi di risposta, sulla scalabilità,
       sull'usabilità, sulla protezione dei dati, etc...
          - etichettiamo il requisito in modo da rendere evidente la categoria
          - alcuni requisiti potrebbero influenzare diversi aspetti del sistema,
             nel caso indicarli tutti, insieme anche al tipo di impatto
             (es: portano ad un peggioramento)
    - #text(blue)[le funzionalità coinvolte]: indicare le funzionalità
       che vengono influenzate dal requisito


- Si può predisporre una tabella

#cfigure("images/2024-03-22-10-55-24.png", 100%)

==== Esempio

- Tabella Vincoli per il Villaggio Turistico

#cfigure("images/2024-03-22-10-55-56.png", 100%)


==== Analisi delle Interazioni

- Vanno distinte le interazioni con gli #text(blue)[umani]
    da quelle con #text(blue)[sistemi esterni]
- Nel caso delle interazioni con gli umani
    - analizzare le eventuali interfacce identificate con il cliente
       nella fase di Analisi dei Requisiti, oppure delineare
       le possibili interfacce
    - individuare le maschere di inserimento/output dati
    - individuare le sole informazioni necessarie da mostrare
       in ogni maschera
    - creare un legame tra maschere - informazioni - funzionalità


- Si può predisporre una tabella

#cfigure("images/2024-03-22-10-59-16.png", 100%)


==== Esempio

- Tabella Maschere per il Villaggio Turistico
    
#cfigure("images/2024-03-22-11-00-16.png", 100%)

- Nel caso delle interazioni con sistemi esterni
    - analizzare ai morsetti i sistemi esterni
       con cui si dovrà interagire
    - individuare i protocolli di interazione con tali sistemi

#cfigure("images/2024-03-22-11-04-11.png", 100%)

==== Esempio

- Tabella Sistemi Esterni per il Villaggio Turistico

#cfigure("images/2024-03-22-11-06-45.png", 100%)

=== Analisi Ruoli e Responsabilità

- Per ogni Attore individuato nei Casi d'Uso specificare
    - le responsabilità - cosa deve fare in ogni funzionalità
    - specificare le #text(blue)[informazioni a cui può accedere]
       relativamente a ciascuna funzionalità in cui è coinvolto
       con relativa indicazione del #text(blue)[tipo di accesso]
    - le maschere che può visualizzare
    - il suo livello di riservatezza - quale livello di riservatezza
       è necessario avere per poter ricoprire quel ruolo
    - la numerosità attesa - il numero di persone che possono giocare
       quel ruolo


====  Analisi dei Ruoli e Responsabilità

- Si possono predisporre alcune tabelle

#cfigure("images/2024-03-22-11-08-17.png", 100%)

Come prima, una sola tabella per tutti i ruoli e una tabella per ogni ruolo.

Non ha senso specificare una numerosità potenzialmente infinita, perché il numero delle persone che interagiranno con il sistema è una informazione che influenza come il sistema verrà progettato. Se la numerosità cambia nel tempo, probabilmente servirà riprogettare il sistema. È possibile inserire come specifica di numerosità "si progetti il sistema in modo che possa supportare il maggior numero di utenti".

==== Esempio

- Tabella Ruoli per il Villaggio Turistico

#cfigure("images/2024-03-22-11-09-38.png", 100%)

==== Esempio

- Tabella Operatore-Informazioni per il Villaggio Turistico

#cfigure("images/2024-03-22-11-09-58.png", 100%)

=== Scomposizione del Problema

- Il punto di partenza è l'Analisi delle Funzionalità
- Per ogni funzionalità marcata come “#text(blue)[complessa]”
    nella Tabella delle Funzionalità occorre valutare
    se sia possibile operare una scomposizione
       - la funzionalità può essere suddivisa in sotto-funzionalità
          più semplici?
       - quale “legame” sussiste tra le sotto-funzionalità?
       - quali informazioni devono “fluire” tra le sotto-funzionalità


- Si possono predisporre alcune tabelle

#cfigure("images/2024-03-22-15-58-43.png", 100%)
La tabella Sotto-Funzionalità elenca le dipendenze tra funzionalità.

- Tabella Scomposizione Funzionalità
    per il Villaggio Turistico

#cfigure("images/2024-03-22-15-59-45.png", 65%)

- Tabella Sotto-Funzionalità per il Villaggio Turistico

#cfigure("images/2024-03-22-16-00-46.png", 100%)

=== Creazione Modello del Dominio

- Il punto di partenza è costituito dall'insieme di:
    - #text(blue)[glossario] definito nell'Analisi dei Requisiti
    - tabelle informazioni/flusso
- Sulla base di questi si costruisce il diagramma
    delle classi che rappresenta il Modello del Dominio
- Tale modello poi sarà riusato nell'Architettura Logica
    come modello dei dati

- Occorre tenere presente che non tutti i vocaboli
    elencati nel glossario diventeranno classi,
    si devono infatti evitare:
       - ridondanze: classi uguali ma con diverso nome
       - costruzione di classi a partire da termini ambigui
          nel glossario
       - nomi che si riferiscono a eventi o operazioni
       - nomi appartenenti al meta-linguaggio del processo
          come, per esempio, requisiti e sistema
       - nomi al di fuori dell'ambito del sistema
       - nomi che possono essere attributi


- Individuare
    - #text(blue)[Oggetti] e #text(blue)[classi] rilevanti per il problema
       che si sta analizzando
          - Limitarsi esclusivamente a quelle classi che fanno parte
             del vocabolario del dominio del problema, non dell'applicazione in se, quindi evitare, ad esempio, di pensare a "che tipo di classe Java devo sviluppare"
    - #text(blue)[Relazioni tra le classi]
    - Per ogni classe
       - #text(blue)[Attributi]
       - #text(blue)[Operazioni fondamentali]
          cioè servizi forniti all'esterno

Pensare sempre al "_cosa_" non al "_come_"; il "come" non è l'oggetto di questa fase.
#cfigure("images/2024-03-22-16-02-37.png", 100%)
Questo schema non è per forza in successione, nel senso che se, mentre stiamo definendo le operazioni di una classe, mi accorgo che c'è bisogno di un attributo, è possibile aggiungerlo.


==== Individuazione delle Classi

- Dal glossario eliminare i nomi che sicuramente
    - non si riferiscono a classi
    - indicano attributi (dati di tipo primitivo)
    - indicano operazioni
- Scegliere un solo termine significativo
    se più parole indicano lo stesso concetto (sinonimi)
- Il nome della classe deve essere un nome familiare
    - all'utente o all'esperto del dominio del problema
    - non allo sviluppatore!

- #text(blue)[Attenzione agli aggettivi e agli attributi], possono
    - indicare oggetti diversi
    - indicare usi diversi dello stesso oggetto
    - essere irrilevanti
- Ad esempio:
    - “Studente bravo” potrebbe essere irrilevante
    - “Studente fuori corso” potrebbe essere una nuova classe
- #text(blue)[Attenzione alle frasi passive, impersonali
    o con soggetti fuori dal sistema]:
       - devono essere rese attive ed esplicite,
          perché potrebbero mascherare entità rilevanti
          per il sistema in esame


- Individuare #text(blue)[Attori] con cui il sistema in esame
    deve interagire
       - #text(blue)[Persone]: Docente, Studente, Esaminatore, Esaminando, ...
       - #text(blue)[Sistemi esterni]: ReteLocale, Internet, DBMS, ...
       - #text(blue)[Dispositivi]: attuatori, sensori, ...
- Individuare #text(blue)[modelli] e #text(blue)[loro elementi specifici]:
    - Insegnamento - “Ingegneria del Software T”
    - CorsoDiStudio -“Ingegneria Informatica”
    - Facoltà - “Ingegneria”
- Individuare #text(blue)[cose tangibili], cioè oggetti reali appartenenti
    al dominio del problema
       - Banco, LavagnaLuminosa, Schermo, Computer, ...

- Individuare #text(blue)[contenitori] (fisici o logici) di altri oggetti
    - Facoltà, Dipartimento, Aula, SalaTerminali, ...
    - ListaEsame, CommissioneDiLaurea, OrdineDegliStudi, ...
- Individuare #text(blue)[eventi] o #text(blue)[transazioni] che il sistema
    deve gestire e memorizzare
       - possono avvenire in un certo istante (es., una vendita) o
       - possono durare un intervallo di tempo (es., un affitto)
       - Appello, EsameScritto, Registrazione, AppelloDiLaurea, ...

- Per determinare se includere una classe nel modello,
    porsi le seguenti domande:
       - il sistema deve interagire in qualche modo
          con gli oggetti della classe?
             - #text(blue)[utilizzare informazioni] (attributi) contenute negli oggetti
                della classe
             - #text(blue)[utilizzare servizi] (operazioni) offerti dagli oggetti della classe
       - quali sono le responsabilità della classe
          nel contesto del sistema?

- Attributi e operazioni devono essere applicabili
    a tutti gli oggetti della classe
- Se esistono
    - attributi con un valore ben definito solo per alcuni oggetti
       della classe
    - operazioni applicabili solo ad alcuni oggetti della classe
siamo in presenza di #text(blue)[ereditarietà]
- Esempio: dopo una prima analisi, la classe Studente
    potrebbe contenere un attributo booleano inCorso,
    ma un'analisi più attenta potrebbe portare alla luce
    la gerarchia:
       - Studente: StudenteInCorso, StudenteFuoriCorso



==== Individuazione delle Classi: Villaggio Turistico

#cfigure("images/2024-03-22-16-06-33.png", 100%)
- Villaggio Turistico è in verde perché è una classe che non viene creata, dato che non serve; se, invece, avessimo avuto un progetto che prevedeva molteplici villaggi turistici, allora avrebbe avuto senso definire la classe `VillaggioTuristico`
- I nomi in rosso sono delle classi, tra l'altro con una probabile gerarchia (alcune)
- Forse `Stanza` e `StanzaCollegata` sono una gerarchia 
#cfigure("images/2024-03-22-16-07-27.png", 100%)

==== Individuazione delle Relazioni

- La maggior parte delle classi (degli oggetti) #text(blue)[interagisce]
    con altre classi (altri oggetti) in vari modi
- In ogni tipo di relazione, esiste un cliente C che dipende
    da un fornitore di servizi F
       - C ha bisogno di F per lo svolgimento di alcune funzionalità
          che C non è in grado di effettuare autonomamente
       - Conseguenza
          per il corretto funzionamento di C
          è indispensabile il corretto funzionamento di F

#cfigure("images/2024-03-22-16-08-12.png", 100%)
- Nell'associazione il fornitore è il contenuto e il cliente è il contenitore, perché il contenitore contenendo il contenuto può usare i servizi del contenuto

==== Individuazione dell'Ereditarietà

- L'ereditarietà deve rispecchiare una tassonomia
    effettivamente presente nel dominio del problema
       - Non usare l'ereditarietà dell'implementazione
          (siamo ancora in fase di analisi!)
       - Non usare l'ereditarietà solo per riunire
          caratteristiche comuni
             - ad es., Studente e Dipartimento hanno entrambi un indirizzo,
                ma non per questo c'è ereditarietà!


==== Ereditarietà: Villaggio Turistico

#cfigure("images/2024-03-22-16-11-00.png", 100%)
- Tutte le ereditarietà di questo caso sono complete e disgiunte; cioè non esistono ospiti che non siano paganti e non paganti e un ospite è o pagante o non pagante, non esistono ospiti non paganti che non siano minorenni e maggiorenni e un ospite non pagante è o minorenne o maggiorenne

====  Individuazione delle Associazioni

- Un'associazione rappresenta una relazione strutturale
    tra due istanze di classi diverse o della stessa classe
- Un'associazione può
    - Rappresentare un contenimento logico (#text(blue)[aggregazione])
       - Una lista d'esame contiene degli studenti
    - Rappresentare un contenimento fisico (#text(blue)[composizione])
       - Un triangolo contiene tre vertici
    - Non rappresentare un reale contenimento
       - Una fattura si riferisce a un cliente
       - Un evento è legato a un dispositivo


===  Individuazione delle Associazioni

- #text(blue)[Aggregazione]\
  Un oggetto x di classe X è associato a (contiene) un oggetto y di classe Y in modo non esclusivo x può condividere y con altri oggetti anche di tipo diverso (che a loro volta possono contenere y)

- Una lista d'esame contiene degli studenti
    - Uno studente può essere contemporaneamente
       in più liste d'esame
    - La cancellazione della lista d'esame non comporta
       l'eliminazione “fisica” degli studenti in lista

- #text(blue)[Composizione]\
    Un oggetto x di classe X è associato a (contiene)
    un oggetto y di classe Y #text(blue)[in modo esclusivo]
    y esiste solo in quanto contenuto in x
- Un triangolo contiene tre punti (i suoi vertici)
    - L'eliminazione del triangolo comporta l'eliminazione
       dei tre punti
- Se la distruzione del contenitore comporta la distruzione
    degli oggetti contenuti, si tratta di composizione,
    altrimenti si tratta di aggregazione

- Attenzione alle associazioni molti a molti
    possono nascondere una classe
    (classe di associazione) del tipo “evento da ricordare”
- Ad esempio,
    - la connessione “matrimonio” tra Persona e Persona
       può nascondere una classe Matrimonio, che lega due Persone
    - la connessione “possiede” tra Proprietario e Veicolo
       può nascondere una classe CompraVendita,
       che lega un Proprietario a un Veicolo


==== 1 ° Esempio di Associazione

#cfigure("images/2024-03-22-16-13-31.png", 100%)
#cfigure("images/2024-03-22-16-13-49.png", 100%)
Quindi, quando l'associazione ha delle informazioni, o più raramente del comportamento, si introduce la classe d'associazione.

==== 2 ° Esempio di Associazione

#cfigure("images/2024-03-22-16-14-47.png", 100%)
- Qui c'è un vincolo particolare, che è quello di tenere in conto del caso in cui una persona compra un veicolo da solo e, successivamente, vuole averlo in comproprietà
- Da notare che ci può essere solo un'istanza della classe di associazione per ogni coppia di oggetti associati

==== Associazioni: Villaggio Turistico


#cfigure("images/2024-03-22-16-15-30.png", 100%)
#cfigure("images/2024-03-22-16-15-47.png", 100%)

==== Individuazione Collaborazioni

- Una classe A è in relazione USA con una classe B
    (A USA B) quando A ha bisogno della collaborazione
    di B per lo svolgimento di alcune funzionalità che A
    non è in grado di effettuare autonomamente
       - Un'operazione della classe A ha bisogno come argomento
          di un'istanza della classe B
             - `voidfun1(B b) { ... usa b ... }`
       - Un'operazione della classe A restituisce un valore di tipo B
          - `B fun2(...) { B b; ... return b; }`
       - Un'operazione della classe A utilizza un'istanza della
          classe B (ma non esiste un'associazione tra le due classi)
             - `void fun3(...) { B b = new B(...); ... usa b ... }`


- La relazione non è simmetrica: A dipende da B, ma B non dipende da A
- Evitare situazioni in cui una classe, tramite una catena
    di relazioni USA, alla fine dipende da se stessa

#cfigure("images/2024-03-22-16-17-13.png", 100%)

==== Individuazione degli Attributi

- Ogni attributo modella una proprietà atomica
    di una classe
       - Un valore singolo
          - Una descrizione, un importo, ..
       - Un gruppo di valori strettamente collegati tra loro
          - Un indirizzo, una data, ..
- Proprietà non atomiche di una classe devono essere
    modellate come associazioni
- A tempo di esecuzione, in un certo istante,
    ogni oggetto avrà un valore specifico per ogni attributo
    della classe di appartenenza: informazione di stato

- La base di partenza per la ricerca degli attributi
    sono le Tabelle di Informazione/Flusso
- Il nome dell'attributo
    - deve essere un nome familiare
       - all'utente o all'esperto del dominio del problema
       - non allo sviluppatore!
    - non deve essere il nome di un valore
       (“qualifica” sì, “ricercatore” no)
    - deve iniziare con una lettera minuscola
- Esempi
    - cognome, dataDiNascita, annoDiImmatricolazione

- Esprimere tutti i vincoli applicabili all'attributo
    - Tipo (semplice o enumerativo)
    - Opzionalità
    - Valori ammessi
       - dominio, anti-dominio, univocità
    - Vincoli di creazione
       - valore iniziale di default, immodificabilità del valore
          (readonly), etc.
    - Vincoli dovuti ai valori di altri attributi
    - Unità di misura, precisione

- Esprimere tutti i vincoli applicabili all'attributo
    - #text(blue)[Visibilità] (opzionale in fase di analisi)
       Attenzione: #text(blue)[gli attributi membro devono essere sempre
       privati!]
    - Appartenenza alla classe (e non all'istanza)
       Attributi (e associazioni) possono essere di classe,
       cioè essere unici nella classe
       - numIstanze: int = 0
- I vincoli possono essere scritti
    - Utilizzando direttamente UML
    - Utilizzando Object Constraint Language (OCL)
    - Come testo in formato libero in un commento UML

#cfigure("images/2024-03-22-16-18-49.png", 100%)

- Attenzione: nel caso di attributi con valore booleano
    “vero o falso”, “sì o no”, il nome dell'attributo
    potrebbe essere uno dei valori di un'enumerazione
       - Ad esempio: tassabile (sì o no) potrebbe diventare
          tipoTassazione{ tassabile, esente, ridotto, ... }
- Attenzione: attributi
    - con valore “non applicabile” o
    - con valore opzionale o
    - a valori multipli (enumerazioni)
    possono nascondere ereditarietà o una nuova classe

#cfigure("images/2024-03-22-16-19-21.png", 100%)

- Attenzione: nel caso di attributi calcolabili (ad esempio,
    età), specificare sempre l'operazione di calcolo
    mai l'attributo
       - se memorizzare oppure no un attributo calcolabile
          è una decisione progettuale,
          un compromesso tra tempo di calcolo e spazio di memoria
- Applicare l'ereditarietà:
    - Posizionare attributi e associazioni più generali
       più in alto possibile nella gerarchia
    - Posizionare attributi e associazioni specializzati
       più in basso



==== Individuazione degli Attributi: Villaggio Turistico

#cfigure("images/2024-03-22-16-20-18.png",100%)
#cfigure("images/2024-03-22-16-20-58.png",100%)


- GuestCard una classe, l'identificativo sarà mappato con una associazione

- Saldo nasconde una nuova classe, associata alle funzionalità di Apertura e Chiusura Credito →L'ospite ha un conto aperto presso il Villaggio Turistico
- Movimento è relativo all'uso della GuestCard

#cfigure("images/2024-03-22-16-24-20.png",100%)



==== Individuazione delle Operazioni

- Il nome dell'operazione
    - deve appartenere al vocabolario standard del dominio
       del problema
    - potrebbe essere un verbo
       - all'imperativo (scrivi, esegui, ...) o
       - in terza persona (scrive, esegue, ...)
    - in modo consistente in tutto il sistema

- Operazioni standard
    - Operazioni che tutti gli oggetti hanno
       per il semplice fatto di esistere e di avere degli attributi
       e delle relazioni con altri oggetti (costruttore, accessori, ...)
    - Sono implicite e, di norma, non compaiono nel diagramma
       delle classi di analisi
- Altre operazioni
    - Devono essere determinate
       - servizi offerti agli altri oggetti
    - Compaiono nel diagramma delle classi di analisi

- Classi contenitori
    - Operazioni standard - aggiungi, rimuovi, conta, itera, ...
    - Altre operazioni - riguardano l'insieme degli oggetti,
       non il singolo oggetto
          - Calcoli da effettuare sugli oggetti contenuti
             Es: calcolaSulleParti(), totalizza()
          - Selezioni da fare sugli oggetti contenuti
             Es: trovaPartiSpecifiche()
          - Operazioni del tipo
             Es: eseguiUnAzioneSuTutteLeParti()

- Distribuire in modo bilanciato le operazioni nel sistema
- Mettere ogni operazione “vicino” ai dati
    a essa necessari
- Applicare l'ereditarietà
    - Posizionare le operazioni più generali più in alto possibile
       nella gerarchia
    - Posizionare le operazioni specializzate più in basso
- Descrivere tutti i vincoli applicabili all'operazione
    - Parametri formali, loro tipo e significato
    - Pre-condizioni, post-condizioni, invarianti di classe
    - Eccezioni sollevate
    - Eventi che attivano l'operazione
    - Applicabilità dell'operazione
    - ...

- PRE-CONDIZIONE
    Espressione logica riguardante le aspettative sullo stato
    del sistema prima che venga eseguita un'operazione
       - Esplicita in modo chiaro che è responsabilità della procedura
          chiamante controllare la correttezza degli argomenti passati
       - Ad esempio, per l'operazione
          *`CalcolaRadiceQuadrata(valore)`* ,
          la pre-condizione potrebbe essere: “ *`valore ≥ 0`* ”

- POST-CONDIZIONE
    Espressione logica riguardante le aspettative sullo stato
    del sistema dopo l'esecuzione di un'operazione
       - Ad esempio, per l'operazione
          *CalcolaRadiceQuadrata(valore)* ,
          la post-condizione potrebbe essere:
          `valore == risultato * risultato`

- #text(red)[INVARIANTE di classe]\
    Vincolo di classe (espressione logica) che deve essere
    sempre verificato
       - sia all'inizio
       - sia alla fine
di tutte le operazioni pubbliche della classe
Può non essere verificato solo durante l'esecuzione
dell'operazione


==== Individuazione delle Operazioni

- ECCEZIONE
    Si verifica quando un'operazione
       - viene invocata nel rispetto delle sue pre-condizioni
       - ma non è in grado di terminare la propria esecuzione
          nel rispetto delle post-condizioni


==== Esempio

- Modello del Dominio “Vendita Servizi” per il Villaggio Turistico


==== Esempio

- Modello del Dominio “Ospite” per il Villaggio Turistico


==== Esempio

- Modello del Dominio “Pagamenti” per il Villaggio Turistico


==== Esempio

- Modello del Dominio “Log” per il Villaggio Turistico






==== Architettura Logica: Struttura

- La parte strutturale dell'Architettura Logica dovrebbe
    essere composta di due tipi differenti di diagrammi UML
       - Diagramma dei Package
          che fornisce una visione di alto livello dell'architettura
       - Diagramma delle classi (uno o più diagrammi in base
          alla complessità) che fornisce una visione più dettagliata
          del contenuto dei singoli package
- Sarebbe opportuno organizzare sin da subito
    l'Architettura Logica usando un pattern architetturale

chiamato Boundary-Control-Entity (BCE)


==== BCE

- BCE è un pattern architetturale che suggerisce di basare
    l'architettura di un sistema sulla partizione sistematica
    degli use case in oggetti di tre categorie:
       - informazione
       - presentazione
       - controllo
- A ciascuna di queste dimensioni corrisponde
    uno specifico insieme di classi
- Tale pattern è stato introdotto anche in RUP
    e sono state adottate icone ben particolari


==== BCE

- BCE è un pattern architetturale che suggerisce di basare
    l'architettura di un sistema sulla partizione sistematica
    degli use case in oggetti di tre categorie:
       - informazione
       - presentazione
       - controllo
- A ciascuna di queste dimensioni corrisponde
    uno specifico insieme di classi
- Tale pattern è stato introdotto anche in RUP
    e sono state adottate icone ben particolari


Noi useremo una
convenzione più
semplice →coloriamo
package e classi in
modo diverso


==== BCE

- _Entity:_ è la dimensione relativa alle entità
    cui corrisponde l'insieme delle classi che includono
    funzionalità relative alle informazioni
    che caratterizzano il problema
       - costituiscono gran parte del modello del dominio
- _Boundary:_ è la dimensione relativa alle funzionalità
    che dipendono dall'ambiente esterno cui corrisponde
    l'insieme delle classi che incapsulano l'interfaccia
    del sistema verso il mondo esterno
- _Control: è_ la dimensione relativa agli enti
    che incapsulano il controllo
       - il loro compito èdi fare da _collante_ tra le interfacce e le entità


==== BCE

- Impostare l'architettura di un sistema software
    distinguendo tra _boundary_ , _control_ ed _entity_
    costituisce un solido punto di partenza
    per l'organizzazione dell'Architettura Logica
    di molte applicazioni
- L'analista tende ad affrontare la complessità
    dei problemi partizionando i problemi stessi
    in sotto-problemi
- È del tutto logico che un analista eviti di associare
    alle entità di un dominio
       - sia le funzionalità di una specifica applicazione
       - sia le funzionalità tipiche della interazione con l'utente


==== BCE

- La conseguenza è che l'architettura
    di un sistema software risulta quasi fisiologicamente
    articolata in una sequenza di livelli ( _layer_ ) verticali
    che viene tipicamente mantenuta
    anche in fase di progetto e implementazione


==== Layer

- Naturale separare la parte che realizza
    le entità dell'applicazione (dominio)
    dalla parte che realizza l'interazione con l'utente (logica
    di presentazione), introducendo una parte centrale
    di connessione (control)


==== Layer

- Nello specifico:
    - il livello di *_presentazione_*
       comprende le parti che realizzano l'interfaccia utente
          - Per aumentare la riusabilità delle parti, questo livello èprogettato
             e costruito astraendo quanto più possibile
             dai dettagli degli specifici dispositivi di I/O
    - il livello di *_applicazione_*
       comprende le parti che provvedono a elaborare
       l'informazione di ingresso, a produrre i risultati attesi
       e a presentare le informazioni in uscita
    - il livello delle *_entità_*
       forma il (modello del) dominio applicativo


==== Struttura: Package

- Usando come base di partenza il lavoro di analisi
    svolto nelle fasi precedenti
    e tenendo in considerazione il pattern BCE
    si può iniziare la creazione del Diagramma dei Package
    che rappresenta la visione dal alto livello
    dell'Architettura Logica


==== Struttura: Package

- Il primo package che si può identificare
    è il package costituito dal Modello del Dominio
    creato nella fase precedente
       - tale Modello (se ben realizzato) costituisce la parte “entity”
          dell'architettura
- Poi è possibile creare un package
    per ognuna delle diverse funzionalità identificate
    nella Tabella delle Funzionalità (“control”)
- Vengono creati uno o più package
    per la parte di “boundary”
- Infine si identificano le _dipendenze logiche_ tra i package


==== Struttura: Package


==== Struttura: Package


Presentazione


Applicazione


Entità


==== Struttura: Package


Le dipendenze
indicate sono solo
a scopo di
esempio. Le
dipendenze, sono
di solito suggerite
dal problema
stesso!

Interazioni con
sistemi / API
Interazioni con utenti esterni


==== Esempio

- Diagramma dei Package per il Villaggio Turistico


uses


uses uses


uses


uses
uses uses uses uses uses uses uses


uses
uses


==== Struttura: Classi

- Dopo aver stabilito la struttura di alto livello
    dell'Architettura Logica
    si dettagliano le classi che compongono ogni package
- Il Package del Dominio è già stato identificato
    nel Modello del Dominio
- Se le classi da specificare per ogni package sono poche
    si può realizzare un unico diagramma delle classi
- Altrimenti è possibile creare un diagramma delle classi
    separato per ogni package


==== Struttura: Classi

- Attenzione a non introdurre scelte di progettazione
    in questa fase
- Indicare solo quelle classi che sono deducibili
    dal problema
       - in genere viene inserita almeno una classe
          che realizza le funzionalità indicate nelle specifiche
       - se nel diagramma dei package è stata indicata
          una funzionalità che nella fase precedente era stata
          poi scomposta si possono indicare le classi
          che realizzano le sotto-funzionalità
       - indicare le classi che rappresentano le maschere
          di interazione con l'utente identificate nelle fasi precedenti


==== Esempio

- Diagramma delle classi per il Villaggio Turistico
    - Diagramma delle classi: InterfacciaCommesso&
       GestionePuntoVendita


==== Esempio

- Diagramma delle classi per il Villaggio Turistico
    - Diagramma delle classi: InterfacciaLog& Log


HomeLog


ViewAnomalie


uses


uses


uses ViewLog
LogController
+ getEntry(DateTime,DateTime)+ getEntry(Date)
+ getAnomalieOperazioni (Date)+ getAnomalieMessaggi(Date)

AnomalieOperazioni


+ scanOperazioni(Date)

AnomalieMesssaggi


+ scanMessaggi(D ate)


=== Architettura Logica: Interazione





==== Architettura Logica: Interazione

- Descrivere le interazioni tra le entità identificate
    nella parte strutturale attraverso opportuni
    Diagrammi di Sequenza
- Diagrammi di sequenza - evidenziano
    - lo scambio di messaggi (le interazioni) tra gli oggetti
    - l'ordine in cui i messaggi vengono scambiati tra gli oggetti
       (sequenza di invocazioni delle operazioni)


==== Architettura Logica: Interazione

- Non spingere la definizione dei diagrammi di sequenza
    sino ai minimi dettagli
- Utilizzare i diagrammi solo per descrivere
    il funzionamento del sistema
       - in risposta a sollecitazioni esterne
       - in fasi particolarmente significative
       - nei casi più critici


==== Esempio

- Diagramma di sequenza per il Villaggio Turistico


=== Architettura Logica: Comportamento





====  Architettura Logica: Comportamento

- Descrivere il comportamento delle entità
    identificate nella parte strutturale attraverso opportuni
    Diagrammi di Stato o delle Attività
- Diagramma di Stato per mostrare come si comportano
    alcune entità complesse a seguito delle interazioni
    e degli eventi che avvengono nel sistema
- Diagramma delle Attività per dettagliare funzionamenti
    complessi delle entità
- Non spingere la definizione dei diagrammi
    sino ai minimi dettagli


- Lo stato di un oggetto è dato dal valore dei suoi attributi
    e delle sue associazioni
- In molti domini applicativi, esistono oggetti che,
    a seconda del proprio stato,
    rispondono in maniera diversa ai messaggi ricevuti
       - Dispositivi (spento, in attesa, operativo, guasto, ecc.)
       - Transazioni complesse (in definizione, in esecuzione,
          completata, fallita, ecc.)
- In questi casi, è opportuno disegnare
    un diagramma di stato per l'oggetto,
    mostrando i possibili stati e gli eventi
    che attivano transizioni da uno stato all'altro

====  Architettura Logica: Comportamento


- A un oggetto possono essere assegnate responsabilità
    che comportano un insieme di elaborazioni complesse
    che devono essere eseguite in un ordine particolare
- In questi casi, è opportuno disegnare
    un diagramma di attività per l'oggetto,
    mostrando le diverse elaborazioni che devono
    essere portate a termine e l'ordine di tali elaborazioni

====  Architettura Logica: Comportamento


==== Esempio

- Diagramma di stato per AnomalieMessaggi


RicezioneMessaggio
entry / do / attesa messaggioelimina protezione
exit /


ScanMessaggio
entry / do / memorizza messaggioelimina protezione
exit /


nuovo mes saggio


AnalizzaMessaggio
entry / do / analizza Etichettaestrai Mittente e Destinatario
exit /


DaDestinatario inviato
entry / do / cerca tra i messaggi dei
Mittenti se già presenteexit /


no
DaMittente
entry / do / cerca tra i messaggi dei
Destinatari se già presenteexit /


sì


presente presente
Memorizza
entry / do / memorizza messaggio
exit /


no no


Confronta
entry / do / confronta messaggi
exit /


sì sì


CreaAnomalia
entry / do / crea nuova AnomaliaMessaggio
exit /


mes sagg i uguali


mes saggi uguali


=== Definizione del Piano di Lavoro





==== Definizione Piano di Lavoro

- Dopo la creazione dell'Architettura Logica
    è possibile iniziare a suddividere il lavoro
- In particolare è necessario:
    - suddividere le responsabilità ai diversi membri
       del team di progetto e sviluppo
    - stabilire le tempistiche per la progettazione di ciascuna parte
    - stabilire i tempi di sviluppo di ciascun sotto-sistema
    - programmare i test di integrazione tra le parti
    - identificare i tempi di rilascio delle diverse versioni del prototipo
    - identificare un piano per gli sviluppi futuri


==== Esempio

*Package* Progetto Sviluppo
Dominio Team progettazione +
Team DB

Team sviluppo A +
Team DB
Gestione Ospite Team progettazione Team sviluppo A
GestionePuntoVendita Team progettazione Team sviluppo B

GestioneCatena Team progettazione Team sviluppo B
Login Team sicurezza Team sviluppo
sicurezza
Log Team sicurezza Team sviluppo
sicurezza
InterfacciaOperatore Team progettazione +
Team grafico

Team sviluppo A +
Team Grafico
InterfacciaCommesso Team progettazione +
Team grafico

Team sviluppo B +
Team Grafico
InterfacciaCatena Team progettazione
+Team grafico

Team sviluppo B +
Team Grafico
InterfacciaLogin Team sicurezza+ Team
grafico

Team sicurezza+ Team
grafico
InterfacciaLog Team sicurezza+ Team
grafico

Team sicurezza+ Team
grafico
GestionePersonale Team progettazione +
Team DB


Team sviluppo A +
Team DB


=== Definizione del Piano del Collaudo





==== Definizione Piano del Collaudo

- Al termine dell'Analisi del Problema, i modelli
    che definiscono il dominio e l'Architettura Logica
    dovrebbero dare sufficienti informazioni su *_cosa_*
    le varie parti del sistema debbano fare senza specificare
    ancora molti dettagli del loro comportamento
- Il “ *_cosa fare_* ” di una parte dovrà comprendere
    anche le forme di interazione con le altre parti
- Lo scopo del *_piano del collaudo_* è cercare di precisare
    il comportamento atteso da parte di una entità
    prima ancora di iniziarne il progetto e la realizzazione
- Focalizzando l'attenzione sulle interfacce delle entità
    e sulle interazioni è possibile impostare scenari
    in cui specificare in modo già piuttosto dettagliato la “risposta”
    di una parte a uno “stimolo” di un'altra parte


==== Definizione Piano del Collaudo

- Lo sforzo di definire nel modo più preciso possibile
    un piano del collaudo di un sistema
    prima ancora di averne iniziato la fase di progettazione
    viene ricompensato da
       - una miglior comprensione dei requisiti
       - un approfondimento nella comprensione dei problemi
       - una più precisa definizione dell'insieme delle funzionalità
          (operazioni) che ciascuna parte deve fornire alle altre
          per una effettiva integrazione nel “tutto” che costituirà
          il sistema da costruire
       - comprendere il significato delle entità e specificarne
          nel modo più chiaro possibile il comportamento atteso


==== Definizione Piano del Collaudo

- Un piano del collaudo va concepito e impostato
    da un punto di vista logico, cercando di individuare
    categorie di comportamenti e punti critici
- In molti casi tuttavia può anche risultare possibile
    definire in modo precoce piani di collaudo
    concretamente eseguibili, avvalendosi di strumenti
    del tipo JUnit/NUnit che sono ormai diffusi
    in tutti gli ambienti di programmazione
- Lo sforzo di definire un piano di collaudo concretamente
    eseguibile promuove uno sviluppo controllato, sicuro
    e consapevole del codice poiché il progettista
    e lo sviluppatore possono verificare subito
    in modo concreto la correttezza di quanto sviluppato


==== JUnit

- JUnit (https://junit.org/junit5/docs/current/user-guide/)
    è un framework per unit-testing per il linguaggio Java
- Dovreste conoscerlo già bene da Fondamenti T- 2


==== NUnit

- NUnit (http://nunit.org/) è un framework per unit-testing
    per tutti i linguaggi .Net
- NUnit è inizialmente derivato da JUnit,
    ma è stato totalmente riscritto dalla versione 3
- Troverete la documentazione e la guida all'installazione
    a
    https://github.com/nunit/docs/wiki/Framework-Release-
    Notes


