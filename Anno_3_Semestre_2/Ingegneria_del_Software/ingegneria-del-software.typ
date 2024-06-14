
#import "@preview/physica:0.9.0": *
#import "@preview/i-figured:0.2.3"
#import "@preview/cetz:0.1.2" 
#import "@preview/xarrow:0.2.0": xarrow


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

#set text(13pt)

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

#let nfigure(img, wth) = figure(image("images/"+img, width: wth))

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
#cfigure("images/image-1710162664687-0.png", 90%)
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
#cfigure("images/image-1710162829556-0.png", 90%)
==== Diagramma dei Package
- Un diagramma dei package è un diagramma che illustra come gli elementi di modellazione sono organizzati in package e le relazioni (dipendenze) tra i package
#cfigure("images/image-1710162883736-0.png", 90%)
- Un diagramma può rappresentare package con diversi livelli di astrazione
==== Dipendenze
- UML permette di rappresentare relazioni che NON sussistono fra istanze nel dominio rappresentato, ma sussistono fra gli elementi del modello UML stesso o fra le astrazioni che tali elementi rappresentano
- Dipendenza:
  è rappresentata da una linea tratteggiata orientata che va da un elemento dipendente (Source) ad uno indipendente(Target)
#cfigure("images/image-1710163145240-0.png", 80%)
- Una dipendenza indica che cambiamenti dell'elemento *indipendente* influenzano l'elemento *dipendente*
  
	- modificano il significato dell'elemento dipendente
	- causano la necessità di modificare anche l'elemento dipendente perché i significati sono dipendenti
- UML mette a disposizione nove diversi tipi di dipendenza, ma per i nostri fini consideriamo quasi sempre <<use>>
#cfigure("images/image-1710163162642-0.png", 50%)
==== Diagramma dei Package
- Tutti i diagrammi che utilizzeremo nella fase di analisi saranno trasformati per venir utilizzati nella fase di progettazione (in analisi definisco il "cosa" e in progettazione definisco il "come")
- Quando si usa il diagramma dei package per definire la parte strutturale dell'architettura logica ricordare sempre che si stanno esprimendo *dipendenze logiche* che sussistono tra le entità del problema
- Non è detto che tali dipendenze rimangano tali anche nella fase di progettazione
- Esempio: in sistema per l'interpretazione di una grammatica si hanno due parti fondamentali
  
	- Lexer→strumento che legge e spezza una sequenza di caratteri in sotto-sequenze dette “token”
	- Parser prende in ingresso i token generati, li analizza e li elabora in base ai costrutti specificati nella grammatica stessa
==== Esempio
#cfigure("images/image-1710163246985-0.png", 90%)
È ovvio che è il Parser a usare il Lexer, quindi il diagramma prodotto nella fase di analisi sarà così
#cfigure("images/image-1710163288049-0.png", 90%)
Ma nel diagramma del progetto è il Lexer a fare il Parser
#cfigure("images/image-1710174515137-0.png", 90%)

== Interfaccia
- Le interfacce forniscono un modo per partizionare e caratterizzare gruppi di proprietà
- Un'interfaccia non deve specificare come possa essere implementata, ma semplicemente quello che è necessario per poterla realizzare
- Le entità che realizzano l'interfaccia dovranno fornire una “*^^vista pubblica^^*”(attributi, operazioni, comportamento osservabile all'esterno) conforme all'interfaccia stessa
- Se un'interfaccia dichiara un attributo, non significa necessariamente che l'elemento che realizza l'interfaccia debba avere quell'attributo nella sua implementazione, ma solamente che esso apparirà così a un osservatore esterno
==== Notazione
#cfigure("images/image-1710163566742-0.png", 90%)
  Il diagramma sopra e il diagramma sotto hanno lo stesso significato

== Diagramma delle Classi
- Un diagramma delle classi descrive il tipo degli oggetti facenti parte di un sistema e le varie tipologie di relazioni statiche tra di essi
- I diagrammi delle classi mostrano anche le proprietà e le operazioni di una classe e i vincoli che si applicano alla classe e alle relazioni tra classi
- Le proprietà rappresentano le caratteristiche strutturali di una classe:
	- sono un unico concetto, rappresentato però con due notazioni molto diverse: attributi e associazioni
	- benché il loro aspetto grafico sia molto differente, concettualmente sono la stessa cosa
==== Esempio
#cfigure("images/image-1710163858337-0.png", 100%)
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
#cfigure("images/image-1710163911503-0.png", 90%)
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
#cfigure("images/image-1710168708671-0.png", 90%)
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
#cfigure("images/image-1710169509255-0.png", 60%)
==== Associazioni
- Le associazioni sono un altro modo di rappresentare le proprietà
- Gran parte dell'informazione che può essere associata a un attributo si applica anche alle associazioni
- Un'associazione è una linea continua che collega due classi, orientata dalla classe sorgente a quella destinazione
- Il nome e la molteplicità vanno indicati vicino all'estremità finale dell'associazione:
- la classe destinazione corrisponde al tipo della proprietà
  
  #cfigure("images/image-1710169568202-0.png", 70%)

- È possibile assegnare un nome all'associazione e anche assegnare dei nomi ai “*ruoli*” svolti da ciascun elemento di un associazione
- Gran parte delle cose ce valgono per gli attributi valgono anche per le associazioni
- Anche nei casi in cui non è strettamente necessario, il ruolo può essere utile per aumentare la leggibilità del diagramma
- Ovviamente quando abbiamo un associazione il tipo della proprietà corrisponde alle classi che sono dall'altra parte della associazione, cioè se guardiamo a la proprietà nella classe A sarà di tipo classe B, e viceversa se guardiamo a classe B la proprietà sarà di tipo classe A

#cfigure("images/image-1710169694503-0.png", 95%)


==== Associazioni Bidirezionali
- Una tipologia di associazione è quella *#text(blue)[bidirezionale (o binaria)]*, costituita da una coppia di proprietà collegate, delle quali una è l'inversa dell'altra
- Il collegamento inverso implica che, se seguite il valore di una proprietà e poi il valore della proprietà collegata, dovreste ritornare all'interno di un insieme che contiene il vostro punto di partenza
  
	#cfigure("images/image-1710169720862-0.png", 75%) 

  la natura bidirezionale dell'associazione è palesata dalle *frecce di navigabilità* aggiunte a entrambi i capi della linea
- Con questa specifica stiamo dicendo all'implementatore che nel progetto deve essere garantita la navigabilità in entrambi i sensi; nel caso non ci sia una specifica di navigabilità la realizzazione è deliberata all'implementatore


==== Associazioni Ternarie
Quando si hanno associazioni ternarie (o che coinvolgono più classi) si introduce il simbolo “#text(blue)[diamante]”
  #cfigure("images/image-1710169756579-0.png", 80%)


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
  
  #cfigure("images/image-1710169956471-0.png", 100%)

- In generale le classi di associazione sono delle classi che sono venute fuori più avanti nel progetto, perché, ad esempio, l'associazione in figura si può realizzare anche con una classe normale
- La classe di associazione aggiunge implicitamente un vincolo extra: ci può essere #underline[solo un'istanza della classe di associazione] tra ogni coppia di oggetti associati



=== Aggregazione e Composizione
- #highlight(fill: myblue)[Aggregazione:]
	- è un associazione che corrisponde a una relazione intuitiva Intero-Parte (“*part-of*”)
	- è rappresentata da un *#text(blue)[diamante vuoto]* sull'associazione, posto vicino alla classe le cui istanze sono gli “interi”
	  #cfigure("images/image-1710170826307-0.png", 40%)
	- è una relazione *binaria*
	- può essere *ricorsiva*
- #highlight(fill: myblue)[Composizione:]
	- è un aggregazione che rispetta due vincoli ulteriori:
		- una parte può essere inclusa in al massimo un intero in ogni istante
		- solo l'oggetto intero può creare e distruggere le sue parti, cioè, le parti non esistono al di fuori dell'intero; questo implica, nell'implementazione, che non esiste un costruttore per le parti
	- è rappresentata da un *#text(blue)[diamante pieno]* vicino alla classe che rappresenta gli “interi”
	  #cfigure("images/image-1710170843379-0.png", 40%)
	- se l'oggetto che compone viene distrutto,anche i figli vengono distrutti, anche se i figli possono essere creati/distrutti in momenti diversi dalla creazione/distruzione dell'oggetto che compone
	- può essere *ricorsiva*
=== Generalizzazione
- La generalizzazione è indicata con una *freccia vuota* fra due classi dette *#text(blue)[sottoclasse]* e *#text(blue)[superclasse]*
- Il significato della generalizzazione è il seguente:
  ogni istanza della sottoclasse è anche istanza della superclasse
  #cfigure("images/image-1710171110210-0.png", 40%)
- La stessa superclasse può partecipare a diverse generalizzazioni
- Una generalizzazione può essere *disgiunta*, cioè le sottoclassi sono disgiunte (non hanno istanze in comune), o meno
  #cfigure("images/image-1710171249961-0.png", 70%)
- Una generalizzazione può essere *#text(blue)[completa]* (l'unione delle istanze delle sottoclassi è uguale all'insieme delle istanze della superclasse), o meno
- *Attenzione*: i valori di default sono `{incomplete, disjoint}`
  #cfigure("images/image-1710171327443-0.png", 75%)
- In una generalizzazione la sottoclasse non solo può avere caratteristiche aggiuntive rispetto alla superclasse, ma può anche *#text(blue)[sovrascrivere]* (*#text(blue)[overriding]*) le proprietà ereditate dalla superclasse
  #cfigure("images/image-1710171353810-0.png", 70%)
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
 #cfigure("images/image-1710419146192-0.png", 80%)
- Questa figura ci dice che ci sono tre modi diversi di vedere una persona:
	- dal punto di vista del sesso
	- dal punto di vista dell'essere paziente o meno
	- dal punto di vista del ruolo
  
==== Relazione tra classi: sintassi

- Attenzione all'uso corretto delle frecce: UML è un linguaggio (anche se grafico) e scambiare una freccia per un'altra è un errore non da poco
  #cfigure("images/image-1710419179129-0.png", 80%)
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
  #cfigure("images/image-1710419317953-0.png", 23%)

== Diagramma di sequenza
- Diagramma che illustra le interazioni tra le classi / entità disponendole lungo una sequenza temporale
- In particolare mostra i soggetti (chiamati tecnicamente *#text(blue)[Lifeline]*) che partecipano all'interazione e la sequenza dei messaggi scambiati
- In ascissa troviamo i diversi soggetti (non necessariamente in ordine di esecuzione), mentre in ordinata abbiamo la scala dei tempi sviluppata verso il basso
#cfigure("images/image-1710419751978-0.png", 100%)
- Le linee della _Lifeline_ partono da un quadrato, un'entità definita con sintassi UML. In questo caso i ":" indicano un'istanza della classe
- Tra la classe `User` e la classe `ACSystem` ci sarà sicuramente un'associazione
- La figura ci die inoltre che i due messaggi che vengono inviati dopo la ricezione di `PIN` possono venire inviati in momenti diversi
=== Esempio
  #cfigure("images/image-1710419777733-0.png", 80%)
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
#cfigure("images/image-1710420323320-0.png", 90%)
*N.B.* Noi non utilizzeremo mai vincoli temporali
#cfigure("images/image-1710420355654-0.png", 100%)

=== Riferimento ad altri Diagrammi
- Spesso i diagrammi di sequenza possono assumereu una certa complessità
	- necessità di poter definire comportamenti più articolati come composizione di nuclei di interazione più semplici
- Oppure, se una sequenza di eventi ricorre spesso, potrebbe essere utile definirla una volta e richiamarla dove necessario
- Per questa ragione, UML permette di inserire #text(blue)[*riferimenti ad altri diagrammi*] e passare loro degli argomenti
- Ovviamente ha senso sfruttare quest'ultima opzione solo se il diagramma accetta dei parametri sui quali calibrare l'evoluzione del sistema
- Questi riferimenti prendono il nome di #text(blue)[InteractionUse]
- I punti di connessione tra i due diagrammi prendono il nome di #text(blue)[Gate]
- Un Gate rappresenta un #text(blue)[punto di interconnessione] che mette in relazione un messaggio al di fuori del frammento di interazione con uno all'interno del frammento
#cfigure("images/image-1710420483529-0.png", 100%)


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
#cfigure("images/image-1710420606015-0.png", 90%)
- Attenzione alle frecce che usate nei messaggi, hanno significati diversi:
	- #highlight(fill: myblue)[riga continua freccia piena:] indica un messaggio (*call*) *sincrono* in cui il mittente *aspetta* il completamento dell'esecuzione del destinatario prima di continuare la sua esecuzione
	  #cfigure("images/image-1710420774542-0.png", 25%)
		- Necessita di un messaggio di ritorno per sbloccare l'esecuzione del mittente
	- #highlight(fill: myblue)[riga continua freccia vuota:] indica un messaggio *asincrono* in cui il mittente *non aspetta* il completamento dell'esecuzione del destinatario ma continua la sua esecuzione
	  #cfigure("images/image-1710420774542-1-1710421153612-0.png", 27%)
		- Il valore di ritorno potrebbe o meno essere necessario, dipende dalla semantica
	- #highlight(fill: myblue)[riga tratteggiata freccia vuota:] indica il ritorno di un messaggio
	  #cfigure("images/image-1710421186142-0.png", 27%)
	
#cfigure("images/image-1710421256766-0.png", 100%)
- Non specifichiamo chi è che ha inviato il messaggio, in questo diagramma non ci interessa
- Register manda il messaggio `doA` all'oggetto `Sale` e aspetta la sua terminazione
- Poi manda il messaggio `doB` all'oggetto `Sale` e aspetta la sua terminazione
- Infine invia il messaggio `doC`, e come conseguenza `Sale` invia `doD` alla classe `Register`
#cfigure("images/image-1710421277921-0.png", 100%)
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
#cfigure("images/image-1710421496294-0.png", 100%)
#cfigure("images/image-1710421508172-0.png", 100%)
Fintanto che ci sono degli `item`, io faccio `enter item` e quello mi restituisce `description total`. Quando non esistono più `item` invio il messaggio `endSale`.
#cfigure("images/image-1710421519605-0.png", 90%)
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
#cfigure("images/image-1710421992238-0.png", 100%)
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
#cfigure("images/image-1710422071247-0.png", 100%)
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
#cfigure("images/image-1710422303917-0.png", 96%)
Questo è uno stato composito, nel senso che quello stato composito può essere inserito all'interno di un altro diagramma di stato. E quindi rappresenta la fattorizzazione che abbiamo visto, è analoga a quello che abbiamo visto nel diagramma dell'attività.


==== Esempio
#cfigure("images/image-1710422325220-0.png", 96%)
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
#cfigure("images/image-1710422403624-0.png", 100%)


==== Pseudostato
- Gli pseudostati vengono generalmente utilizzati per collegare più transizioni in percorsi di transizioni di stato più complessi
- Ad esempio, combinando una transizione che entra in uno pseudostato fork con un insieme di transizioni che escono dallo pseudostato fork, otteniamo una transizione composta che porta a un insieme di stati ortogonali
==== Tipi di Pseudostati
- #highlight(fill: myblue)[Initial:] vertice di default che è la sorgente della singola transizione verso lo stato di default di un composite state - Ci può essere al massimo un vertice initial e la transizione uscente da questo vertice non può avere trigger o guardie
- #highlight(fill: myblue)[deepHistory]: la più recente configurazione attiva del composite state che contiene direttamente questo pseudostato
- #highlight(fill: myblue)[shallowHistory:] il più recente substate attivo di un composite state initial deepHistory shallowHistory
  #cfigure("images/image-1710422587280-0.png", 65%)
- #highlight(fill: myblue)[Join:] permette di eseguire il merge di diverse transizioni provenienti da diverse sorgenti appartenenti a differenti regioni ortogonali
	- Le transizioni entranti in questo vertice non possono avere guardie e trigger
- #highlight(fill: myblue)[Fork:] permette di separare una transizione entrante in due o più transizioni che terminano in vertici di regioni ortogonali
		- Le transizioni uscenti da questo vertice non possono avere guardie
#cfigure("images/image-1710426556871-0.png", 75%)
- #highlight(fill: myblue)[Junction:] è un vertice privo di semantica che viene usato per “incatenare” insieme transizioni multiple
	- È usato per costruire percorsi di transizione composti tra stati
	- Una junction può essere usata per far convergere transizioni multiple entranti in una singola transizione uscente che rappresenta un percorso condiviso di transizione
	- Allo stesso modo una junction può essere usata anche per suddividere una transizione entrante in più transizioni uscenti con differenti guardie
	- Permette di realizzare un #underline[branch condizionale statico], , quindi quando esco da un certo stato, so già in quale delle biforcazioni finirò
	- Esiste una guardia predefinita chiamata “else”che viene attivata quando tutte le guardie sono false
#cfigure("images/image-1710426578024-0.png", 70%)
- #highlight(fill: myblue)[Choice:] è un tipo di vertice che quando viene raggiunto causa la valutazione dinamica delle guardie dei trigger delle transizioni uscenti
	- Le guardie sono quindi tipicamente scritte sotto forma di “funzione”che viene valutata al momento del raggiungimento del vertice choice
	- Permette di realizzare un #underline[branch condizionale dinamico] separando le transizioni uscenti e creando diversi percorsi di uscita
	- Se più di una guardia è verificata viene scelto il percorso in modo arbitrario
	- Se nessuna è verificata il modello viene considerato “ill-formed”
	- È quindi caldamente consigliato definire sempre un percorso di uscita di tipo “else”
#cfigure("images/image-1710426645434-0.png", 70%)
- La differenza tra _Junciton_ e _Choise_ è che, siccome nelle transizioni io posso eseguire delle azioni, la biforcazione di junction non può dipendere dalle azioni, mentre la biforcazione di choice può dipendere dall'esito di quelle azioni.

- #highlight(fill: myblue)[Entry point:] è l'ingresso di uno state machine o di un composite state
- #highlight(fill: myblue)[Exit point:] è l'uscita da uno state machine o da uno stato composito
	- Entrare in questo vertice significa attivare la transizione che ha questo vertice come sorgente
- #highlight(fill: myblue)[Terminate:] entrare in questo vertice implica che l'esecuzione di questo state machine è terminata
	- Lo state machine non uscirà da nessuno stato né verranno invocate altre azioni a parte quelle associate con la transizione che porta allo stato terminate
#cfigure("images/image-1710426724993-0.png", 75%)
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
#cfigure("images/image-1710426803459-0.png", 86%)
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
  #cfigure("images/image-1710427558944-0.png", 25%)
- Attività interna: generazione di un thread concorrente che resta in esecuzione finché:
	- il thread (eventualmente) termina
	- si esce dallo stato attraverso una transizione esterna
#cfigure("images/image-1710427643942-0.png", 25%)

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
#cfigure("images/image-1710427924392-0.png", 100%)
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
#cfigure("images/image-1710428025939-0.png", 100%)
- #highlight(fill: myblue)[Decision e Merge]: determinano il comportamento condizionale
	- #highlight(fill: myblue)[Decision] ha una singola transizione entrante  e più transizioni uscenti in cui solo una di queste sarà prescelta
	- #highlight(fill: myblue)[Merge] ha più transizioni entranti e una sola uscente e serve  a terminare il blocco condizionale cominciato con un decision
- #highlight(fill: myblue)[Fork e join:] determinano il comportamento parallelo:
	- quando scatta la transazione entrante, si eseguono in parallelo  tutte le transazioni che escono dal fork
		- Con il parallelismo non è specificata la sequenza
		- Le fork possono avere delle guardie
	- Per la sincronizzazione delle attività è presente il costrutto di join che ha più transazioni entranti e una sola transazione uscente
#cfigure("images/image-1710428137459-0.png", 40%){:height 343, :width 434}


=== Object Flow
- È un arco che ha oggetti o dati che fluiscono su di esso
#cfigure("images/image-1710428157989-0.png", 100%)


=== Segnali ed Eventi
#cfigure("images/image-1710428183146-0.png", 100%)


=== Exception Handler
#cfigure("images/image-1710428204930-0.png", 100%)

#pagebreak()
=== InterruptibleActivityRegion
#cfigure("images/image-1710428254604-0.png", 100%)





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
- ![image.png](images/image-1710063362882-0.png)
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
- ![image.png](images/image-1710063695471-0.png)
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
 ![image.png](images/image-1710064987414-0.png)
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
- ![image.png](images/image-1710065295316-0.png)
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

Quali sono le minacce?

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

Architettura Client/Server DoS

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


#line(length: 100%)

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

#cfigure("images/2024-03-22-16-18-49.png", 85%)

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

- #highlight(fill: myblue)[Pre-condizione]\
    Espressione logica riguardante le aspettative sullo stato
    del sistema prima che venga eseguita un'operazione
       - Esplicita in modo chiaro che è responsabilità della procedura
          chiamante controllare la correttezza degli argomenti passati
       - Ad esempio, per l'operazione
          *`CalcolaRadiceQuadrata(valore)`* ,
          la pre-condizione potrebbe essere: “ *`valore ≥ 0`* ”

- #highlight(fill: myblue)[Post-condizione]\
    Espressione logica riguardante le aspettative sullo stato
    del sistema dopo l'esecuzione di un'operazione
       - Ad esempio, per l'operazione
          *CalcolaRadiceQuadrata(valore)* ,
          la post-condizione potrebbe essere:
          `valore == risultato * risultato`

- #highlight(fill: myred)[Invariante di classe]\
    Vincolo di classe (espressione logica) che deve essere
    sempre verificato
       - sia all'inizio
       - sia alla fine
di tutte le operazioni pubbliche della classe
Può non essere verificato solo durante l'esecuzione
dell'operazione



- ECCEZIONE
    Si verifica quando un'operazione
       - viene invocata nel rispetto delle sue pre-condizioni
       - ma non è in grado di terminare la propria esecuzione
          nel rispetto delle post-condizioni


==== Esempio

Modello del Dominio “Vendita Servizi” per il Villaggio Turistico
#cfigure("images/2024-03-26-12-01-43.png",100%)

Modello del Dominio “Ospite” per il Villaggio Turistico
#cfigure("images/2024-03-26-12-02-40.png", 100%)

Modello del Dominio “Pagamenti” per il Villaggio Turistico
#cfigure("images/2024-03-26-12-03-09.png", 100%)

Modello del Dominio “Log” per il Villaggio Turistico
#cfigure("images/2024-03-26-12-03-33.png", 100%)





=== Architettura Logica: Struttura

*N.B.* Le parti "Struttura", "Interazione" e "Comportamento" dell'architettura logica vanno sviluppate contestualmente (in contemporanea?).

- La parte strutturale dell'Architettura Logica dovrebbe
    essere composta di due tipi differenti di diagrammi UML
       - #text(blue)[Diagramma dei Package] \
          che fornisce una visione di alto livello dell'architettura
       - #text(blue)[Diagramma delle classi] \ (uno o più diagrammi in base
          alla complessità) che fornisce una visione più dettagliata
          del contenuto dei singoli package
- Sarebbe opportuno organizzare sin da subito
    l'Architettura Logica usando un pattern architetturale chiamato #text(blue)[*Boundary-Control-Entity*] (BCE)


==== BCE

- BCE è un pattern architetturale che suggerisce di basare
    l'architettura di un sistema sulla partizione sistematica
    degli use case in oggetti di tre categorie:
       - #highlight(fill: myred)[informazione]
       - #highlight(fill: myred)[presentazione]
       - #highlight(fill: myred)[controllo]

#cfigure("images/2024-03-26-12-06-04.png", 65%)
Noi useremo una convenzione più semplice → coloriamo package e classi in modo diverso

- A ciascuna di queste dimensioni corrisponde
    uno specifico insieme di classi
- Tale pattern è stato introdotto anche in RUP
    e sono state adottate icone ben particolari

- BCE è un pattern architetturale che suggerisce di basare
    l'architettura di un sistema sulla partizione sistematica
    degli use case in oggetti di tre categorie:
       - informazione
       - presentazione
       - controllo


- #highlight(fill: myblue)[_Entity:_] è la dimensione relativa alle entità
    cui corrisponde l'insieme delle classi che includono
    funzionalità relative alle informazioni
    che caratterizzano il problema; sostanzialmente ci dice quali sono le entità di interesse nel dominio del problema
       - costituiscono gran parte del modello del dominio
- #highlight(fill: myblue)[_Boundary:_] è la dimensione relativa alle funzionalità
    che dipendono dall'ambiente esterno cui corrisponde
    l'insieme delle classi che incapsulano l'interfaccia
    del sistema verso il mondo esterno
- #highlight(fill: myblue)[_Control:_] è la dimensione relativa agli enti
    che incapsulano il controllo
       - il loro compito è di fare da _collante_ tra le interfacce e le entità


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

- La conseguenza è che l'architettura
    di un sistema software risulta quasi fisiologicamente
    articolata in una sequenza di livelli (_layer_) verticali
    che viene tipicamente mantenuta
    anche in fase di progetto e implementazione


==== Layer

- È naturale separare la parte che realizza
    le entità dell'applicazione (dominio)
    dalla parte che realizza l'interazione con l'utente (logica
    di presentazione), introducendo una parte centrale
    di connessione (control)


- Nello specifico:
    - il livello di #text(blue)[*_presentazione_*]
       comprende le parti che realizzano l'interfaccia utente
          - Per aumentare la riusabilità delle parti, questo livello èprogettato
             e costruito astraendo quanto più possibile
             dai dettagli degli specifici dispositivi di I/O
    - il livello di #text(blue)[*_applicazione_*]
       comprende le parti che provvedono a elaborare
       l'informazione di ingresso, a produrre i risultati attesi
       e a presentare le informazioni in uscita
    - il livello delle #text(blue)[*_entità_*]
       forma il (modello del) dominio applicativo


==== Struttura: Package

- Usando come base di partenza il lavoro di analisi
    svolto nelle fasi precedenti
    e tenendo in considerazione il pattern BCE
    si può iniziare la creazione del Diagramma dei Package
    che rappresenta la visione dal alto livello
    dell'Architettura Logica


- Il primo package che si può identificare
    è il package costituito dal #text(blue)[Modello del Dominio]
    creato nella fase precedente
       - tale Modello (se ben realizzato) costituisce la parte “#text(blue)[entity]”
          dell'architettura
- Poi è possibile creare un package
    per ognuna delle diverse funzionalità identificate
    nella Tabella delle Funzionalità (“control”); ciascuna delle righe della tabella diventa un package di controllo
- Vengono creati uno o più package
    per la parte di “boundary”
- Infine si identificano le _dipendenze logiche_ tra i package


#cfigure("images/images/2024-03-26-13-48-53-png.png", 90%)
#cfigure("images/2024-03-26-13-49-53.png", 90%)


#pagebreak()

==== Esempio

Diagramma dei Package per il Villaggio Turistico

#cfigure("images/2024-03-26-13-50-45.png", 100%)


==== Struttura: Classi

- Dopo aver stabilito la struttura di alto livello
    dell'Architettura Logica
    #text(red)[si dettagliano le classi che compongono ogni package]
- Il Package del Dominio è già stato identificato
    nel Modello del Dominio
- Se le classi da specificare per ogni package sono poche
    si può realizzare un unico diagramma delle classi
- Altrimenti è possibile creare un diagramma delle classi
    separato per ogni package

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
    - Diagramma delle classi: `InterfacciaCommesso` &
       `GestionePuntoVendita`

#cfigure("images/2024-03-26-13-52-51.png", 90%)

- Diagramma delle classi per il Villaggio Turistico
    - Diagramma delle classi: `InterfacciaLog` & `Log`

#cfigure("images/2024-03-26-13-53-41.png", 100%)




=== Architettura Logica: Interazione


- Descrivere le interazioni tra le entità identificate
    nella parte strutturale attraverso opportuni
    Diagrammi di Sequenza
- I #text(blue)[Diagrammi di sequenza] evidenziano
    - lo scambio di messaggi (le interazioni) tra gli oggetti
    - l'ordine in cui i messaggi vengono scambiati tra gli oggetti
       (sequenza di invocazioni delle operazioni)

- Non spingere la definizione dei diagrammi di sequenza
    sino ai minimi dettagli
- Utilizzare i diagrammi solo per descrivere
    il funzionamento del sistema
       - in risposta a sollecitazioni esterne
       - in fasi particolarmente significative
       - nei casi più critici

- Non serve mostrare tutti i diagrammi di interazione, ma magari fare vedere quelli più interessanti e/o complicati

==== Esempio

Diagramma di sequenza per il Villaggio Turistico

#cfigure("images/2024-03-26-13-56-06.png", 100%)

*N.B.* Mai fare vedere la password nel modello del dominio. L'interfaccia che gestisce il login avrà un auto-anello, quindi solo lei dovrà gestire il controllo password; #underline[come] realizzare questa cosa si vedrà in progettazione. 

=== Architettura Logica: Comportamento

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

- A un oggetto possono essere assegnate responsabilità
    che comportano un insieme di elaborazioni complesse
    che devono essere eseguite in un ordine particolare
- In questi casi, è opportuno disegnare
    un diagramma di attività per l'oggetto,
    mostrando le diverse elaborazioni che devono
    essere portate a termine e l'ordine di tali elaborazioni


==== Esempio

Diagramma di stato per `AnomalieMessaggi`

#cfigure("images/2024-03-26-13-57-51.png", 100%)

=== Definizione del Piano di Lavoro


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

#cfigure("images/2024-03-26-13-58-40.png", 100%)


=== Definizione del Piano del Collaudo

- Al termine dell'Analisi del Problema, i modelli
    che definiscono il dominio e l'Architettura Logica
    dovrebbero dare sufficienti informazioni su #text(blue)[*_cosa_*]
    le varie parti del sistema debbano fare senza specificare
    ancora molti dettagli del loro comportamento
- Il “#text(blue)[*_cosa fare_*]” di una parte dovrà comprendere
    anche le forme di interazione con le altre parti
- Lo scopo del #text(blue)[*_piano del collaudo_*] è cercare di precisare
    il comportamento atteso da parte di una entità
    prima ancora di iniziarne il progetto e la realizzazione
- Focalizzando l'attenzione sulle interfacce delle entità
    e sulle interazioni è possibile impostare scenari
    in cui specificare in modo già piuttosto dettagliato la “risposta”
    di una parte a uno “stimolo” di un'altra parte


- Lo sforzo di definire nel modo #text(red)[più preciso possibile]
    un piano del collaudo di un sistema
    prima ancora di averne iniziato la fase di progettazione
    viene ricompensato da
       - una #text(blue)[miglior comprensione] dei requisiti
       - un approfondimento nella #text(blue)[comprensione dei problemi]
       - una più precisa definizione dell'insieme delle funzionalità
          (operazioni) che ciascuna parte deve fornire alle altre
          per una #text(blue)[effettiva integrazione] nel “tutto” che costituirà
          il sistema da costruire
       - comprendere il #text(blue)[significato delle entità] e specificarne
          nel modo più chiaro possibile il #text(blue)[comportamento atteso]


==== Definizione Piano del Collaudo

- Un piano del collaudo va concepito e impostato
    da un punto di #text(blue)[_*vista logico*_], cercando di individuare
    categorie di comportamenti e punti critici
- In molti casi tuttavia può anche risultare possibile
    definire in modo precoce #text(blue)[piani di collaudo
    concretamente eseguibili], avvalendosi di strumenti
    del tipo JUnit/NUnit che sono ormai diffusi
    in tutti gli ambienti di programmazione
- Lo sforzo di definire un piano di collaudo concretamente
    eseguibile promuove uno sviluppo #text(red)[controllato, sicuro
    e consapevole] del codice poiché il progettista
    e lo sviluppatore possono verificare subito
    in modo concreto la correttezza di quanto sviluppato


==== JUnit

- JUnit (https://junit.org/junit5/docs/current/user-guide/)
    è un framework per unit-testing per il linguaggio Java
- Dovreste conoscerlo già bene da Fondamenti T-2


==== NUnit

- NUnit (http://nunit.org/) è un framework per unit-testing
    per tutti i linguaggi .Net
- NUnit è inizialmente derivato da JUnit,
    ma è stato totalmente riscritto dalla versione 3
- Troverete la documentazione e la guida all'installazione
    a
    https://github.com/nunit/docs/wiki/Framework-Release-Notes




== Progettazione
=== Introduzione
- #text(red)[Obiettivo:] attraverso una serie di raffinamenti successivi dell'Architettura Logica arrivare ad ottenere #text(blue)[l'Architettura del Sistema]\
#text(blue)[Vanno considerati anche tutti gli aspetti vincolanti che sono stati trascurati nelle fasi precedenti]
    - Questa fase deve mirare non solo a individuare e descrivere una soluzione al problema (#text(red)[what/how]), ma soprattutto a descrivere i #text(blue)[motivi] (#text(red)[why]) che l'hanno determinata

- #text(red)[Risultato]:
    - Architettura del Sistema
    - Schema Persistenza
    - Piano finale del Collaudo
    - Indicazioni per il Deployment


#cfigure("images/2024-04-26-12-52-24.png",15%)




+ #text(blue)[*Progettazione Architetturale*]\
    #underline[Obiettivo:] definire l'Architettura del Sistema tenendo conto di tutti i vincoli e delle forze in gioco
+ #text(blue)[*Progettazione di Dettaglio*]\
    #underline[Obiettivo:] progettare nel dettaglio ogni aspetto del Sistema
+ #text(blue)[*Progettazione della Persistenza*]\
    #underline[Obiettivo:] progettare i meccanismi per la persistenza dei dati
+ #text(blue)[*Progettazione del Collaudo*]\
    #underline[Obiettivo:] definire in modo chiaro e preciso come il sistema dovrà essere collaudato una volta terminata l'implementazione
+ #text(blue)[*Progettazione per il Deployment*]\
    #underline[Obiettivo:] progettare il sistema in modo da rendere semplice il deployment sulle macchine e per garantire la sicurezza


#cfigure("images/2024-04-26-12-55-41.png",100%)

=== Progettazione Architetturale
- Nella Progettazione Architetturale gli ingegneri devono prendere delle decisioni che influenzano profondamente il sistema
- Basandosi sulle proprie esperienze e conoscenze devono rispondere ad alcune domande fondamentali:
    + C'è un'architettura applicativa generica che può essere utilizzata come modello per il sistema che sto progettando?
    + Come sarà distribuito il sistema tra più processori?
    + Quale stile o quali stili sono adatti al sistema?
    + Quale sarà l'approccio fondamentale utilizzato per strutturare il sistema?
    + Come saranno scomposte in moduli le unità strutturali del sistema?
    + Quale strategia sarà usata per controllare l'operato delle unità del sistema?


#cfigure("images/2024-04-26-12-57-50.png",100%)

==== Requisiti Non Funzionali

- L'architettura del sistema influenza
    - le #highlight(fill: myred)[prestazioni]
    - la #highlight(fill: myred)[robustezza]
    - la #highlight(fill: myred)[distribuibilità]
    - la #highlight(fill: myred)[manutenibilità]
    di un sistema

- La struttura dell'architettura tipicamente è condizionata
    - dalla #text(blue)[tipologia di applicazione] che si vuole realizzare

    - dai #text(blue)[requisiti non funzionali]


- Se le #text(blue)[*prestazioni*] sono un requisito critico l'architettura dovrebbe essere progettata
    - localizzando le operazioni critiche all'interno di un piccolo numero di componenti
    - minimizzando le comunicazioni possibile tra essi

- Questo porta a dover definire componenti “#text(blue)[grandi]” per ridurre la comunicazione


- Se la #text(blue)[*protezione dei dati (_security_)*] è un requisito critico l'architettura dovrebbe essere progettata
    - con una struttura “stratificata”
    - collocando le risorse più critiche nello strato più interno e protetto
- Questo porta a dover definire una struttura con un alto livello di convalida di protezione a ogni strato
- *NB:* Quando si valuta l'aspetto della protezione dei dati tenere conto di tutte le indicazioni che sono emerse nella parte della Security Engineering


- Se la sicurezza (safety) è un requisito critico l'architettura dovrebbe essere progettata
    - in modo tale che le operazioni relative siano tutte collocate in un singolo componente o in un piccolo insieme di componenti
    - riduzione dei costi e dei problemi di convalida della sicurezza, possibilità di poter fornire sistemi di protezione correlati
- Questo porta a dover definire componenti “#text(blue)[grandi]” per localizzare le operazioni




- Se la #text(blue)[*disponibilità*] è un requisito critico l'architettura dovrebbe essere progettata
    - per comprendere componenti ridondanti
    - in modo che sia possibile sostituirli e aggiornarli senza fermare il sistema

- Questo porta a dover sviluppare un numero maggiore di componenti rispetto a quelli strettamente necessari




- Se la #text(blue)[*manutenibilità*] è un requisito critico l'architettura dovrebbe essere progettata
    - usando componenti piccoli, atomici, autonomi
    - che possano essere modificati velocemente
    - i produttori di informazione dovrebbero essere separati dai consumatori e le strutture dati condivise dovrebbero essere evitate
- Questo porta a dover sviluppare componenti di piccole dimensioni


- Ci sono dei conflitti potenziali tra alcune di queste architetture così come abbiamo visto sussistono conflitti tra i requisiti non funzionali

- Esempio: usare componenti “grossi” migliora le prestazioni ma peggiora la manutenibilità e viceversa
- Se sono entrambi requisiti critici occorre trovare un compromesso


==== Esempio

Nell'Analisi del Problema (Tabella Vincoli) sono emersi tre requisiti non funzionali che
impongono dei vincoli al sistema:

- Tempo di risposta
- Usabilità
- Sicurezza

Nello specifico caso in esame, Usabilità e Sicurezza hanno pochi conflitti a parte l'eventuale
richiesta di un ulteriore login se per caso scade la sessione di lavoro. L'Usabilità impatta molto
di più la struttura delle interfacce che andranno progettate in modo tale da mantenere nelle
stesse View le informazioni necessarie alle funzionalità richieste
Diversa la questione che riguarda Tempo di risposta e Sicurezza, aggiungere strati (layer) e
meccanismi di cifratura per migliorare la sicurezza ovviamente porta ad un peggioramento
delle prestazioni del sistema, occorre quindi trovare un bilanciamento tra i due aspetti.
Considerando la tipologia di sistema che deve essere sviluppato, si ritiene maggiormente
critico l'aspetto di sicurezza dei dati in quanto la “Tabella Valutazione Beni” mette in luce che
nel caso di attacchi al sistema andati a buon fine si rischia un'esposizione molto alta con
perdite finanziarie e di immagine. Inoltre, gli utenti principali di tale sistema sono operatori
umani che spesso non sono in grado di percepire se il Sistema impiega qualche frazione di
secondo in più o in meno nella risposta, non si hanno vincoli real-time da soddisfare.


==== Scelta Architettura

- La scelta dell'Architettura del Sistema deve basarsi su:
    - Architettura Logica definita in fase di Analisi del Problema
    - Trade-off requisiti non funzionali
    - Tipologia di applicazione che si intende sviluppare
    - Adozione di Pattern Architetturali
        - Blackboard
        - MVC/BCE
        - Layers
        - Client/Server
        - Broker
        - Pipe & Filters
        - ...


===== Blackboard

- Il pattern Blackboard aiuta a strutturare quelle applicazioni in cui vengono applicate strategie di soluzione non deterministiche (tipici problemi di intelligenza artificiale)

- I diversi sotto-sistemi condividono la stesse conoscenze attraverso la Blackboard al fine di costruire una soluzione approssimata o parziale

#cfigure("images/2024-04-26-13-09-07.png",80%)

===== MVC

- Il pattern MVC divide le applicazioni in tre distinte parti:

    - Il model che gestisce i dati

    - Il controller che manipola i dati

    - La view che mostra i dati

#cfigure("images/2024-04-26-13-10-16.png",90%)

===== Layer

- Il pattern Layer aiuta a strutturare quelle applicazioni che possono essere scomposte in gruppi di sotto-attività in cui ciascun gruppo si trova a un ben definito livello di astrazione

#cfigure("images/2024-04-26-13-10-44.png",20%)

===== Client/Server

- Il pattern client/server aiuta a strutturare un'applicazione come un insieme di servizi forniti da uno o più server e un insieme di client che utilizza tali servizi

#cfigure("images/2024-04-26-13-11-26.png",90%)

Tre diverse possibilità:
#cfigure("images/2024-04-26-13-11-54.png",90%)

===== Broker

- Il pattern Broker può essere usato per strutturare sistemi distribuiti con un disaccoppiamento tra i diversi sotto-sistemi che comunicano tra loro attraverso remote server invocation

- Il Broker è responsabile della coordinazione delle comunicazioni,come inoltro richieste, invio risposte ed eccezioni

#cfigure("images/2024-04-29-16-49-33.png",100%)

===== Pipe & Filters

- Il pattern Pipe & Filters aiuta a strutturare quelle applicazioni che processano flussidi dati

- Ogni passo del processo è incapsulato in un apposito filtro e i dati attraversano una pipe di filtri

- Variando l'ordine dei filtri si possono ottenere diversi tipi di sistemi

#cfigure("images/2024-04-26-13-12-43.png",90%)


===== Conclusioni

- Come non esiste un processo di sviluppo ideale, non esiste un'Architettura ideale sempre utilizzabile

- Talvolta è necessario usare stili architetturali diversi per parti diverse del sistema al fine di soddisfare tutti i vincoli imposti dai requisiti

- L'adozione dei pattern architetturali può aiutare a trovare il giusto compromesso tra tutte le forze in gioco


===== Esempio: Villaggio Turistico
#cfigure("images/2024-04-26-13-13-35.png",100%)

==== Scelte Tecnologiche

- L'uso di una specifica tecnologia (intesa anche come linguaggio di programmazione, piattaforma, strumento, etc.) non è sempre neutro

- In taluni casi potrebbe risultare vantaggioso scegliere le tecnologie già in fase di progettazione legando così il progetto alla specifica tecnologia

- Nel caso si decida di scegliere la tecnologia in fase di progettazione, va specificato chiaramente e va fatta un'analisi costi/benefici

- Vanno attentamente studiate le parti della tecnologia adottata in modo che sia poi possibile inserirle nei diagrammi di progettazione


=== Progettazione di dettaglio

- La Progettazione di Dettaglio definisce il dettaglio dell'Architettura del Sistema nelle sue tre viste:

    - Struttura

    - Interazione

    - Comportamento
- Quindi non più architettura logica, ma architettura di sistema

- Per realizzare un sistema funzionante, occorre considerare GUI, DB, Framework, librerie, componenti, modifiche al modello per avere #text(blue)[software estensibile e modulare...]

- È compito della Progettazione di Dettaglio #text(blue)[identificare e definire altre classi] in accordo alla specifica architettura scelta, siamo arrivati al _come_


- Durante la Progettazione di Dettaglio, i modelli prodotti nell'Analisi devono essere #text(blue)[estesi] al fine di progettare i quattro layer principali che compongono il sistema

    - #highlight(fill: myblue)[Application Logic] - logica dell'applicazione e controllo degli altri componenti

    - #highlight(fill: myblue)[Presentation logic] - gestione dell'interazione con l'utente a livello logico nuovi oggetti: finestre, menù, bottoni, toolbar , ...

    - #highlight(fill: myblue)[Data logic] - gestione dei dati che il sistema deve manipolare

    - #highlight(fill: myblue)[Middleware] - gestione dell'interazione con i sistemi esterni, con la rete e tra i sotto-sistemi

- Durante la Progettazione di Dettaglio, i modelli di Analisi devono essere #text(blue)[modificati] al fine di:

    - definire in dettaglio le classi e delle loro relazioni

    - supportare #text(blue)[caratteristiche specifiche] per comunicazioni,

    - diagnostica, protezione dei dati,...

    - #text(blue)[riuso] di classi e/o componenti disponibili

    - miglioramento delle #text(blue)[prestazioni]

    - supporto alla #text(blue)[portabilità]

    - ...


- #text(blue)[*Massima indipendenza possibile*] da

    - Linguaggio (e ambiente) di programmazione

    - DBMS

    - Sistema Operativo

    - Hardware

- Le caratteristiche specifiche del contesto utilizzato devono essere tenute in conto solo se
    - #text(blue)[sono vincolanti] (requisiti non funzionali)

    - si è #text(blue)[esplicitamente scelto] di legarsi a una tecnologia nella progettazione architetturale, ad esempio una specifica tecnologia utilizzabile solo da un linguaggio di programmazione

#cfigure("images/2024-04-26-13-17-41.png",90%)

==== Architettura: Struttura

- Durante la Progettazione di Dettaglio della parte di Struttura è necessario definire

    - #text(blue)[tipi di dato] che non sono stati definiti in precedenza

    - #text(blue)[navigabilità delle associazioni] tra classi

    - #text(blue)[strutture dati] necessarie per l'effettiva implementazione del sistema

    - #text(blue)[operazioni] che non erano emerse durante la fase di Analisi del Problema

    - eventuali #text(blue)[nuove classi] necessarie per il corretto funzionamento del sistema



- Attenzione alla presenza dei “Sistemi Esterni” individuati in fase di Analisi del Problema

- Se nella tabella “Tabella dei Sistemi Esterni” era stato individuato un problema nel “Livello di Protezione” e il Sistema Esterno non risulta avere il livello di sicurezza minimo richiesto occorre applicate il #text(red)[*pattern Adapter*]

    - si ingloba (wrappa) il Sistema Esterno in una nostra struttura

    - si progetta la struttura in modo tale che soddisfi i livelli minimi di sicurezza richiesti



- Attenzione se si è deciso di vincolarsi a una specifica tecnologia

- Va condotta una #text(blue)[attenta analisi] e #text(red)[valutazione del livello di protezione offerto] dalla tecnologia scelta

- Se tale livello non risulta essere quello minimo richiesto dall'applicazione occorre progettare specifiche parti del sistema per prevenire i buchi di sicurezza legati alla specifica tecnologia

- Ove possibile cercare di applicare il pattern Adapter



- Applicazione dei #text(red)[*design pattern*] al fine di realizzare #text(blue)[*software di qualità*] facilmente estensibile e modulare

- Applicazione dei #text(red)[principi di progettazione] con particolare attenzione al #text(red)[“Dependency Inversion Principle”]

- Disaccoppiare i layer del sistema porta molti vantaggi

    - possibile cambiare implementazione di parti del sistema senza che la modifica si ripercuota sulla restante parte: #text(red)[*design for change*]

    - possibile cambiare l'aspetto grafico anche variando la tecnologia realizzativa senza dover modificare l'application logic

    - facile inserire nuove funzionalità con impatto minimo sul sistema


==== Struttura: Esempio

#cfigure("images/2024-04-26-13-22-07.png",100%)
#cfigure("images/2024-04-26-13-22-26.png",100%)
#cfigure("images/2024-04-26-13-22-43.png",100%)
#cfigure("images/2024-04-26-13-23-04.png",100%)

#cfigure("images/2024-04-26-13-23-22.png",50%)

#cfigure("images/2024-04-26-13-23-57.png",100%)


==== Architettura: Interazione

- Durante la Progettazione di Dettaglio della parte di Interazione è necessario

    - #text(blue)[ridefinire i protocolli di interazione] emersi in fase di Analisi dettagliandoli tenendo conto delle nuove entità emerse in progettazione, quindi definiamo il _come_

    - #text(blue)[progettare accuratamente] i protocolli di interazione verso i sistemi esterni

    - #text(blue)[definire nuovi protocolli di interazione] tra le classi che sono state introdotte nella progettazione


===== Interazione: Esempio
#cfigure("images/2024-04-26-13-25-50.png",100%)
#cfigure("images/2024-04-26-13-31-15.png",100%)
#cfigure("images/2024-04-26-13-31-32.png",100%)






=== Architettura: Comportamento

- Durante la progettazione di dettaglio della parte di Comportamento è necessario

- #text(blue)[definire gli algoritmi] che implementano le operazioni complesse/complicate in modo chiaro e preciso avvalendosi eventualmente di diagrammi delle attività

- #text(blue)[dettagliare] i diagrammi di stato/attività già definiti nella fase precedente

- eventualmente #text(blue)[aggiungere diagrammi] di stato/attività per le nuove entità emerse in questa fase


=== Progettazione della persistenza

- La persistenza dei dati è un fattore cruciale nello sviluppo di un sistema

- Il progettista dopo un'attenta valutazione di 
    - vincoli imposti dai requisiti funzionali (tempi di risposta, requisiti di protezione e privacy, ...)

    - tipologia di accesso accesso ai dati (lettura, scrittura, ricerche)

    - frequenza di accesso ai dati (quanto spesso devo accedere ai dati?)

    - criticità e consistenza dei dati (quanto spesso cambiano i dati? quali sono i costi di eventuale “perdite” nelle modifiche dei dati?)
    dovrà scegliere la tecnica migliore di persistenza

- Per ogni sistema va valutato attentamente quale strategia dà il miglior bilanciamento tra i vincoli e le forze in gioco nel sistema

- Non è sempre detto che l'adozione di un (R)DBMS sia la risposta corretta

- Per esempio se dobbiamo memorizzare dei log la strategia migliore è quella delle scrittura su file:
    - la maggior parte delle funzionalità “scrivono” solamente una o più righe nel log e l'accesso deve essere molto veloce: il log non deve pesare troppo nei tempi di risposta nel sistema

    - solo gli strumenti di analisi accedono in lettura al loge solitamente occorre analizzare ogni singola riga nel corretto ordine temporale, non c'è bisogno di fare ricerche


==== Quando usare un DBMS

- In generale possiamo affermare che quando si ha a che fare con:

    - gestionali che trattano un numero considerevole di dati anche di natura eterogenea

    - dati che cambiano molto spesso e devono essere costantemente aggiornati

    - la “perdita” di modifiche può essere un problema
    - necessità di ripristino di versioni precedenti a seguito di un malfunzionamento
    #text(blue)[la scelta consigliata è quella di avvalersi di un DBMS]

#line(length: 100%)

- L'output di questa fase può essere rappresentato da:

    - lo schema E-R del DB che dovrà supportare l'applicazione

    - il formato del/i file che dovranno essere scritti/letti dall'applicazione

- Sarebbe bene che sia nel caso di DB che di file ci fosse una #text(red)[piccola analisi del rischio] per capire se

    - il DB è protetto in modo adeguato

    - il/i file necessitano di meccanismi di protezione

- Il punto di partenza di tale analisi sono #text(blue)[i livelli di protezione e privacy richiesti per i diversi dati] che saranno memorizzati


==== Esempio: DB Villaggio Turistico
#cfigure("images/2024-04-26-19-02-08.png",100%)

==== Esempio: log Villaggio Turistico

- Formato file per Log delle operazioni

    `DataOra operazione esecutore`

- Formato file per Log dei messaggi

  `DataOra messaggio protetto invio/ricezione autore`


=== Progettazione del collaudo

- La base di partenza di questa attività è il Piano del Collaudo sviluppato nell'analisi

- Dopo la Progettazione di Dettaglio è possibile scrivere i #text(blue)[test unitari di ciascuna classe]

- Successivamente vanno progettati con cura anche #text(blue)[i test di integrazione] del sistema

- L'output di questa attività è rappresentato dalla Suite completa dei test unitari e di integrazione


=== Progettazione per il deployment

- Seguiremo le linee guida già viste nel blocco della sicurezza:

    + Includere supporto per visionare e analizzare le configurazioni

    + Minimizzare i privilegi di default

    + Localizzare le impostazioni di configurazione

    + Fornire modi per rimediare a vulnerabilità di sicurezza


==== Deployment: Esempio

#cfigure("images/2024-04-26-19-04-46.png",100%)

#underline[Lato server:]

- i server dovranno essere installati su macchine all'interno di una rete privata

- la rete privata dovrà essere opportunamente protetta da un firewall a cifratura di pacchetti

- l'unico punto di contatto verso l'esterno è il Broker





=== Design Pattern


Nel 1977, Christopher Alexander disse:

#align(center)[_«Each pattern describes a problem which occurs over and over again in our environment, and then describes the core of the solution to that problem, in such a way that you can use the solution a million times over, without ever doing it the same way twice»_]

Parlava di costruzioni civili e di città




- La stessa frase è applicabile anche alla progettazione object-oriented

- In questo caso, le soluzioni utilizzeranno
        - oggetti, classi e interfacce
        - invece che pareti e porte...


- #text(blue)[Obiettivi]
    - Risolvere problemi progettuali specifici
    - Rendere i progetti object-oriented più flessibili e riutilizzabili

- Ogni design pattern
    - Cattura e formalizza l'esperienza acquisita nell'affrontare e risolvere uno specifico problema progettuale
    - Permette di riutilizzare tale esperienza in altri casi simili




- Ogni design pattern ha #text(blue)[*quattro elementi essenziali*]

    - un #text(blue)[*nome*] (significativo) - identifica il pattern

    - il #text(blue)[*problema*] - descrive quando applicare il pattern

    - la #text(blue)[*soluzione*] - descrive il pattern, cioè gli elementi che lo compongono (classi e istanze) e le loro relazioni, responsabilità e collaborazioni

    - le #text(blue)[*conseguenze*] - descrivono vantaggi e svantaggi dell'applicazione del pattern e permettono di valutare le alternative progettuali


==== L'importanza dei nomi dei Pattern

- Gli schemi progettuali del software hanno nomi suggestivi:

    - _Observer, Singleton, Strategy ..._
- Perché i nomi sono importanti?
    - Supportano il _chunking_, ovvero fissano il concetto nella nostra memoria e ci aiutano a capirlo
    - Facilitano la comunicazione tra progettisti


==== Classificazione dei Design Pattern

- #highlight(fill: myblue)[Pattern di creazione] (creational pattern)\
    Risolvono problemi inerenti il processo di creazione di oggetti

- #highlight(fill: myblue)[Pattern strutturali] (structural pattern)\
    Risolvono problemi inerenti la composizione di classi o di oggetti

- #highlight(fill: myblue)[Pattern comportamentali] (behavioral pattern)\
    Risolvono problemi inerenti le modalità di interazione e di distribuzione delle responsabilità tra classi o tra oggetti

#cfigure("images/2024-05-03-13-03-37.png",100%)


==== Pattern SINGLETON

- Assicura che una classe abbia una sola istanza  fornisce un punto di accesso globale a tale istanza

- La classe deve:

    - tenere traccia della sua sola istanza
    - intercettare tutte le richieste di creazione, al fine di garantire che nessuna altra istanza venga creata; sostanzialmente viene dato a chi da la richiesta di creazione il riferimento all'istanza 
    - fornire un modo per accedere all'istanza unica


```C public class Singleton

{
    ... attributi membro di istanza ...
    private static Singleton instance = null;
    protected Singleton()
    { inizializzazione istanza }
    public static Singleton GetInstance()
    {
        if(_instance == null)
        _instance = new Singleton();
        return _instance;
    }
    ... metodi pubblici, protetti e privati ...

}```

- Alternativa: classe non istanziabile (`static class`) con soli membri statici

    - `Math`
    - `Convert`
    - ...

- Perché un _singleton_?

- Il _singleton_ #text(blue)[*può implementare 1+ interfacce*]

- Il _singleton_ #text(blue)[*può essere specializzato*] ed è possibile creare nella `GetInstance` un'istanza specializzata che dipende dal contesto corrente


```C public static Singleton GetInstance()
{
if(_instance == null)
_instance = CreateInstance();
return _instance;
}
private static Singleton CreateInstance()
{
    if(...)
        return new SubSingletonA();
    else if(...)
        return new SubSingletonB();
    else
        return new SubSingletonC();

}```

#cfigure("images/2024-05-03-13-07-12.png",50%)

==== Pattern OBSERVER

- #text(blue)[*Contesto*]

    - Talvolta una modifica a un oggetto (il #text(blue)[*soggetto*]) richiede che altri oggetti (#text(blue)[*osservatori*]) siano modificati a loro volta, o quantomeno richiedano di essere notificati se il soggetto viene modificato

    - Questa relazione può essere esplicitamente codificata nel soggetto, ma questo richiede che questo sappia come gli osservatori debbano essere aggiornati

        - si crea accoppiamento tra gli oggetti (#text(blue)[*closely coupled*]) e #text(blue)[*non possono essere facilmente  riusati*] 

#cfigure("images/2024-05-03-13-09-40.png",50%)



- #text(blue)[*Soluzione*]

    - Creare una relazione uno-a-molti più lasca tra un oggetto e gli altri che dipendono da esso

    - Una modifica dell'oggetto farà sì che gli altri #text(blue)[*ricevano una notifica*], consentendo loro di aggiornarsi di conseguenza

#cfigure("images/2024-05-03-13-10-49.png",90%)
- Ogni `Subject` può attaccarsi o staccarsi da un `Observer`
- Con questo pattern il Soggetto da chi sono gli Osservatori, mentre gli Osservatori non sanno chi è il soggetto
===== Esempio Boss-Worker

- È necessario modellare un'interazione tra due componenti
    - un #text(blue)[*Worker*] che effettua un'attività (o lavoro)
    - un #text(blue)[*Boss*] che controlla l'attività dei suoi Worker
- Ogni _Worker_ deve notificare al proprio Boss:
    - quando il lavoro inizia
    - quando il lavoro è in esecuzione
    - quando il lavoro finisce
- Soluzioni possibili:

+ #text(blue)[*class-based*] _callback relationship_

+ #text(blue)[*interface-based*] _callback relationship_

+ #text(blue)[*pattern Observer*] (lista di notifiche)

+ #text(blue)[*delegate-based*] _callback relationship_

+ #text(blue)[*event-based*] _callback relationship_


#heading(level: 6, numbering: none)[Class-based callback relationship]

#cfigure("images/2024-05-03-13-33-18.png",90%)

#heading(level: 6, numbering: none)[Interface-based callback relationship]

#cfigure("images/2024-05-03-13-34-46.png",100%)

#heading(level: 6, numbering: none)[Pattern Observer (lista di notifiche)]

#cfigure("images/2024-05-03-13-35-30.png",100%)


==== Pattern Model / View / Controller (MVC)

- Utilizzato per realizzare le interfacce utenti in _Smalltalk-80_

- Permette di suddividere un'applicazione, o anche la sola interfaccia dell'applicazione, in tre parti

- #text(blue)[*Modello*]: elaborazione/stato
- #text(blue)[*View*]: (o viewport) output
- #text(blue)[*Controller*]: input

#line(length: 100%)

#text(blue)[*Modello*]

- Gestisce un insieme di dati logicamente correlati
- Risponde alle interrogazioni sui dati
- Risponde alle istruzioni di modifica dello stato
- Genera un evento quando lo stato cambia
- Registra (in forma anonima) gli oggetti interessati alla notifica dell'evento
- In Java, deve estendere la classe `java.util.Observable`



#text(blue)[*View*]

- Gestisce un'area di visualizzazione, nella quale presenta all'utente una vista dei dati gestiti dal modello
    - Mappa (parte de) i dati del modello in oggetti visuali
    - Visualizza tali oggetti su un particolare dispositivo di output

- Si registra presso il modello per ricevere l'evento di cambiamento di stato

- In Java, deve implementare l'interfaccia `java.util.Observer`


#text(blue)[*Controller*]

- Gestisce gli input dell'utente (mouse, tastiera, ...)
- Mappa le azioni dell'utente in comandi
- Invia tali comandi al modello e/o alla view che effettuano le operazioni appropriate
- In Java, è un #text(blue)[_listener_]


#cfigure("images/2024-05-03-13-38-55.png",55%)

Con view passiva:
#cfigure("images/2024-05-03-13-39-41.png",55%)
- In questo caso è il Presenter a parlare con il Model
- Posso avere diversi Presenter per lo stesso Model
==== Pattern FLYWEIGHT

- Descrive come condividere oggetti “leggeri” (cioè a granularità molto fine) in modo tale che il loro uso non sia troppo costoso

- Un _flyweight_ è un #text(blue)[*oggetto condiviso*] che può essere utilizzato simultaneamente ed efficientemente da più clienti (del tutto indipendenti tra loro)

- Benché condiviso, #text(blue)[*non deve essere distinguibile da un oggetto non condiviso
*]
- Non deve fare ipotesi sul contesto nel quale opera


- Per assicurare una corretta condivisione, i clienti

    - non devono istanziare direttamente i flyweight

    - ma devono ottenerli esclusivamente tramite una `FlyweightFactory`

```C private DictionaryKeyType, FlyweightType> flyweights;
...
public FlyweightType GetFlyweight(KeyType key)
{
    if(!flyweights.ContainsKey(key))
    {
        flyweights.Add(key, CreateFlyweight(key));
    }
    return flyweights[key];
}```



- Distinzione tra stato intrinseco e stato estrinseco

- #text(blue)[*Stato intrinseco*]:

    - #text(blue)[*Non dipende dal contesto di utilizzo*] e quindi #text(blue)[*può essere condiviso*] da tutti i clienti

    - Memorizzato nel _flyweight_

- #text(blue)[*Stato estrinseco*]

    - #text(blue)[*Dipende dal contesto di utilizzo*] e quindi #text(blue)[*non può essere condiviso*] dai clienti

    - Memorizzato nel cliente o calcolato dal cliente

    - Viene passato al _flyweight_ quando viene invocata una sua operazione


#cfigure("images/2024-05-03-13-43-19.png",100%)


===== Esempio

- Si supponga di usare il pattern flyweight per condividere delle icone tra vari clienti

#cfigure("images/2024-05-03-13-43-59.png",100%)



- Lo #text(blue)[*stato intrinseco*] (memorizzato nel flyweight) comprenderà tutte le informazioni che i clienti devono (e possono) condividere:

    - Nome dell'icona

    - Bitmap dell'icona

    - Dimensioni originali, ...

- Lo #text(blue)[*stato estrinseco*] (memorizzato nel cliente) comprenderà il contesto in cui l'icona dovrà essere disegnata (dipendente dal singolo cliente):

    - Posizione dell'icona

    - Dimensioni richieste, ...




==== Pattern STRATEGY

- Permette di

    - definire un insieme di algoritmi tra loro correlati,

    - incapsulare tali algoritmi in una gerarchia di classi e

    - rendere gli algoritmi intercambiabili

#cfigure("images/2024-05-03-13-46-13.png",90%)


#cfigure("images/2024-05-03-13-47-41.png",90%)

===== Esempio

- #text(blue)[*Allineamento del testo di un paragrafo*]\
    Esistono politiche diverse di allineamento

#cfigure("images/2024-05-03-13-48-58.png",90%)

- #text(blue)[*AlignerBase*]

    - suddivide il testo in linee (`Format`)

    - delega alle sue sottoclassi l'allineamento delle singole linee (`FormatLine`)

- `Paragraph` utilizza i servizi di un _“Aligner”_ specificato dinamicamente run-time

- È possibile realizzare gli “Aligner” utilizzando il pattern #text(blue)[_*flyweight*_]

Esempio


==== Pattern ADAPTER

- Converte l'interfaccia originale di una classe nell'interfaccia (diversa) che si aspetta il cliente

- Permette a classi che hanno interfacce incompatibili di lavorare insieme
- Si inserisce, tra le due classi, l'interfaccia Adapter, che ingloba la classe che deve erogare i servizi, in modo che possa comunicare con la classe che li deve usare
- Si usa quando

    - si vuole riutilizzare una classe esistente 

    - la sua interfaccia non è conforme a quella desiderata

- Noto anche come #text(blue)[_*wrapper*_]
- Si realizza con composizione-delega


#cfigure("images/2024-05-03-13-51-00.png",90%)

===== Esempio
#cfigure("images/2024-05-03-13-51-38.png",80%)




==== Pattern DECORATOR

- Permette di #text(blue)[*aggiungere responsabilità*] a un oggetto dinamicamente

- Fornisce un'#text(blue)[*alternativa flessibile alla specializzazione*]

    - In alcuni casi, le estensioni possibili sono talmente tante che per poter supportare ogni possibile combinazione, si dovrebbe definire un numero troppo elevato di sottoclassi




- TextBox

    - BorderTextBox

    - FilledTextBox

    - VerticalTextBox

    - BorderFilledTextBox

    - BorderVerticalTextBox

    - BorderFilledVerticalTextBox

    - FilledVerticalTextBox

- E se volessi

    - 2 o più bordi

    - Cambiare il font

    - ...



#cfigure("images/2024-05-03-13-53-22.png",90%)


- #text(blue)[*Component*] (*interfaccia o classe astratta*)
    - Dichiara l'interfaccia di tutti gli oggetti ai quali deve essere possibile aggiungere dinamicamente responsabilità

- #text(blue)[*ConcreteComponent*]
    - Definisce un tipo di oggetto al quale deve essere possibile aggiungere dinamicamente responsabilità

- #text(blue)[*Decorator*] (*classe astratta*)
    - Mantiene un riferimento a un oggetto di tipo Component e  definisce un'interfaccia conforme all'interfaccia di Component

- #text(blue)[*ConcreteDecorator*]
    - Aggiunge responsabilità al componente referenziato


#cfigure("images/2024-05-03-13-55-03.png",90%)


==== Ereditarietà Dinamica

- Una sotto-classe deve sempre essere una #text(blue)[*versione più specializzata*] della sua super-classe (o classe base)

- Un buon test sul corretto utilizzo dell'ereditarietà è che sia valido il #text(blue)[*principio di sostituibilità di Liskov*]:\
    _“B è una sotto-classe di A se e solo se ogni programma che utilizzi oggetti di classe A può utilizzare oggetti di classe B senza che il comportamento logico del programma cambi”_

- Perché ciò sia valido, è necessario che:

    - le *pre-condizioni* di tutti i metodi della sotto-classe siano uguali o più deboli

    - le *post-condizioni* di tutti i metodi della sotto-classe siano uguali o più forti

    - ogni metodo ridefinito nella sotto-classe deve mantenere la *semantica* del metodo originale



#cfigure("images/2024-05-03-13-57-52.png",90%)


- Il metodo `Modifica` della classe `ModificatoreDiDimensioni`

    - funziona correttamente su un `Rettangolo`
    - ma NON funziona correttamente su un `Quadrato`

- Quindi non è possibile passare un'istanza di `Quadrato` dove è prevista un'istanza di `Rettangolo`

    - il principio di sostituibilità di Liskovè violato!

- *Conclusione*: un quadrato NON è un rettangolo perché pone dei nuovi vincoli al concetto di rettangolo

- Come possiamo tenere conto di ciò che il rettangolo e il quadrato hanno in comune?


#cfigure("images/2024-05-03-14-14-23.png",90%)

- Cosa intendiamo esattamente per Rettangolo e per Quadrato?

- #text(blue)[*Rettangolo*]: parallelogramma i cui angoli sono retti

- #text(blue)[*Parallelogramma*]: quadrilatero i cui lati opposti sono paralleli tra loro

- #text(blue)[*Quadrilatero*]: poligono avente quattro lati e quattro angoli

    - Quadrilateri notevoli sono il quadrato, il rettangolo, il parallelogramma, il rombo e il trapezio

- #text(blue)[*Poligono*]: figura geometrica limitata da una linea poligonale chiusa

- #text(blue)[*Rombo*]: parallelogramma equilatero in cui gli angoli adiacenti sono diversi tra loro

- #text(blue)[*Quadrato*]: parallelogramma equilatero ed equiangolo


- Cosa intendiamo esattamente per Rettangolo e per Quadrato nella nostra applicazione?

- *Ipotesi*: abbiamo a che fare esclusivamente con parallelogrammi



#text(blue)[*1. Lati e angoli NON sono modificabili*]

    - Definire quattro classi concrete che derivano dalla classe astratta `Parallelogramma` (o implementano `IParallelogramma`): `Rettangolo , Quadrato , Rombo , ParallelogrammaGenerico`

    - Usare una factory che in base ai valori dei lati e degli angoli istanzia un rettangolo (che NON deve avere i lati uguali), un quadrato, un rombo o un parallelogramma generico

#text(blue)[*2. Lati e angoli sono modificabili*]

- Definire un'unica classe concreta `Parallelogramma` le cui istanze possono comportarsi a seconda del loro stato come: un rettangolo, un quadrato, un rombo, o un parallelogramma generico



- Come può un oggetto cambiare comportamento, al cambiare del suo stato?

- #underline[1 possibilità]: #text(blue)[*si cambia la classe dell'oggetto run-time*]

    - nella maggior parte dei linguaggi di programmazione a oggetti, questo non è possibile (inoltre, è meglio che un oggetto non possa cambiare classe durante la sua esistenza)

- la classe di un oggetto deve basarsi sulla sua essenza e non sul suo stato

- #underline[2 possibilità]: #text(blue)[*si utilizza il _pattern_ State*] che usa un #text(blue)[*meccanismo di delega*], grazie al quale l'oggetto è in grado di comportarsi #text(blue)[*come se*] avesse cambiato classe


==== Pattern STATE

#cfigure("images/2024-05-03-14-21-05.png",80%)

- Localizza il comportamento specifico di uno stato e suddivide il comportamento in funzione dello stato

- Le classi concrete contengono la logica di transizione da uno stato all'altro

- Permette anche di emulare l'ereditarietà multipla

#cfigure("images/2024-05-03-14-21-45.png",100%)


==== Pattern COMPOSITE

- Permette di comporre oggetti in una #text(blue)[*struttura ad albero*], al fine di rappresentare una #text(blue)[*gerarchia di oggetti contenitori-oggetti contenuti*]

- Permette ai clienti di #text(blue)[*trattare in modo uniforme oggetti singoli e oggetti composti*]

#cfigure("images/2024-05-03-14-22-54.png",80%)



- #text(blue)[*Component*] (classe astratta)

    - Dichiara l'interfaccia

    - Realizza il comportamento di default

- #text(blue)[*Client*]

    - Accede e manipola gli oggetti della composizione attraverso l'interfaccia di *Component*


- #text(blue)[*Leaf*]

    - Descrive oggetti che non possono avere figli -foglie

    - Definisce il comportamento di tali oggetti

- #text(blue)[*Composite*]

    - Descrive oggetti che possono avere figli -contenitori

    - Definisce il comportamento di tali oggetti

#cfigure("images/2024-05-03-14-24-35.png",70%)

- Il contenitore dei figli deve essere un attributo di #text(blue)[*Composite*] e può essere di qualsiasi tipo (array, lista, albero, tabella hash, ...)

#line(length: 100%)

- #text(blue)[*Riferimento esplicito al genitore*] (parent)

    - Semplifica l'attraversamento e la gestione della struttura

    - L'attributo che contiene il riferimento al genitore e la relativa gestione devono essere posti nella classe #text(blue)[*Component*]

- #text(blue)[*Invariante*]
    - Tutti gli elementi che hanno come _parent_ lo stesso componente devono essere (gli unici) figli di quel componente

        - incapsulare l'assegnamento di parent nei metodi `Add` e `Remove` della classe Composite , oppure

        - incapsulare le operazioni di `Add` e `Remove` nella set dell'attributo _parent_ della classe `Component`




public class Composite : Component

```C{

    ...

    public void Add(Component aChild)

    {

        if(aChild.Parent != null)

            throw new ArgumentException(...);

        _children.Add(aChild);

        aChild._parent = this;

    }

    ...

}```



```C
public class Composite : Component

{

    ...

    public void Remove(Component aChild)

    {

        if(aChild.Parent != this)

            throw new ArgumentException(...);

        if(!_children.Contains(aChild))

            throw new ArgumentException(...);

        _children.Remove(aChild);

        aChild._parent = null;

    }

    ...

}```



```C public class Component

{


    ...
    public Composite Parent
    {
        get { return _parent; }
        set
        {
            if(value != _parent)
            {
                if(_parent != null)
                    _parent.Remove(this);
                if(value != null)
                    value.Add(this);
            }
        }
    }
    ...

}```




- #text(blue)[*Massimizzazione dell'interfaccia `Component`*]

    - Un obiettivo del pattern Composite è quello di fare in modo che il cliente veda solo l'interfaccia di Component: in `Component` devono essere inserite tutte le operazioni che devono essere utilizzate dai clienti

        - nella maggior parte dei casi, Component definisce una realizzazione di default che le sotto classi devono ridefinire

- Alcune di queste operazioni possono essere prive di significato per gli oggetti foglia (`Add` ,`Remove` , ...)



- #text(blue)[*Trasparenza*]
    - Dichiaro tutto al livello più alto, in modo che il cliente possa trattare gli oggetti in modo uniforme ma... #text(blue)[*il cliente potrebbe cercare di fare cose senza senso*], come aggiungere figli alle foglie

- Se scegliamo la trasparenza

    - `Add` e `Remove` devono avere una realizzazione di default che genera un'eccezione

    - dovremmo disporre di un modo per verificare se è possibile aggiungere figli all'oggetto su cui si vuole agire




```C // Il cliente conosce solo Component

Component parent = ComponentFactory.CreateInstance(...);
...
Component child = ComponentFactory.CreateInstance(...);
...
// Prima di inserire un figlio,
// occorre controllare se è possibile

if(parent.IsComposite())
    parent.Add(child);
```


- #text(blue)[*Sicurezza*]
- Tutte le operazioni sui figli vengono messe in `Composite` - a questo punto, qualsiasi invocazione sulle foglie genera un errore in fase di compilazione ma... #text(blue)[*il cliente deve conoscere e gestire due interfacce differenti*]

- Se scegliamo la sicurezza
    - dobbiamo disporre di un modo per verificare se l'oggetto su cui si vuole agire è un `Composite`



```C // Il cliente conosce Component e Composite

Component child = ComponentFactory.CreateComponent(...);
Composite parent1 = ComponentFactory.CreateComposite(...);
parent1.Add(child);
...
Component parent2 = ComponentFactory.CreateComponent(...);
// Errore di compilazione
parent2.Add(child);
// Prima di inserire un figlio,
// occorre controllare se è possibile e fare un cast

if(parent2 is Composite)
    ((Composite) parent2).Add(child);
```

==== Pattern VISITOR

- Permette di #text(blue)[*definire una nuova operazione*] da effettuare su gli elementi di una struttura, #text(blue)[*senza dover modificare le classi degli elementi coinvolti*]
- Ad esempio, si consideri la rappresentazione di un programma come “#text(blue)[*abstract syntax tree*]” (AST) - i cui nodi descrivono elementi sintattici del programma

- Su tale albero devono poter essere effettuate molte operazioni di tipo diverso

    - Controllare che tutte le variabili siano definite

    - Eseguire delle ottimizzazioni

    - Generare il codice macchina

    - Stampare l'albero in un formato leggibile

    - ...




Per l'AST utilizziamo il _pattern Composite_

#cfigure("images/2024-05-03-18-17-58.png",90%)


- In seguito potremmo voler effettuare #text(blue)[*altri tipi di operazioni*]

    - controllare che le variabili siano state inizializzate prima dell'uso

    - ristrutturare automaticamente il programma

    - calcolare varie metriche

    - ...

- Se distribuiamo le operazioni sui vari tipi di nodo, otteniamo un sistema che è difficile da

    - capire

    - modificare

    - estendere




- La soluzione è quella di eliminare le singole operazioni dall'AST (la cui responsabilità principale è quella di rappresentare un programma sotto forma di albero)
- #text(blue)[*Tutto il codice relativo ad un singolo tipo di operazione*] (ad es., generazione del codice) viene raccolto in #text(blue)[*una singola classe*]
- I nodi dell'AST devono #text(blue)[*accettare la visita*] delle istanze di queste nuove classi (#text(blue)[_*visitor*_])
- Per aggiungere un #text(blue)[*nuovo tipo di operazione*], è sufficiente progettare una #text(blue)[*nuova classe*]




- Il Visitor deve dichiarare #text(blue)[*un'operazione per ogni tipo di nodo*] concreto

#cfigure("images/2024-05-03-18-20-32.png",100%)



- Ogni nodo deve dichiarare #text(blue)[*un'operazione per accettare un generico _visitor_*]

#cfigure("images/2024-05-03-18-21-23.png", 90%)

#line(length: 100%)

- #text(blue)[*Visitor*] (classe astratta o interfaccia)
    - Dichiara un metodo #text(blue)[`Visit`] per ogni classe di elementi concreti

- #text(blue)[*ConcreteVisitor*]
    - Definisce tutti i metodi #text(blue)[`Visit`]

    - Globalmente #text(blue)[*definisce l'operazione da effettuare sulla struttura*] e (se necessario) ha un proprio stato

#cfigure("images/2024-05-03-18-23-17.png",100%)


- #text(blue)[*Element*] (classe astratta o interfaccia)
    - Dichiara un metodo #text(blue)[`Accept`] che accetta un Visitor come argomento

- #text(blue)[*ConcreteElement*]
    - Definisce il metodo #text(blue)[`Accept`]

#cfigure("images/2024-05-03-18-24-20.png",90%)


- #text(blue)[*ObjectStructure*]
    - Può essere realizzata come Composite o come normale collezione (array, lista, ...)
    - Deve poter enumerare i suoi elementi
    - Deve dichiarare un'interfaccia che permetta a un cliente di far visitare la struttura a un Visitor

#cfigure("images/2024-05-03-18-25-00.png",50%)

#line(length: 100%)

- #text(blue)[*Facilita l'aggiunta di nuove operazioni*]
    - È possibile aggiungere nuove operazioni su una struttura esistente, semplicemente aggiungendo un nuovo visitor concreto
    - Senza il pattern Visitor, sarebbe necessario aggiungere un metodo ad ogni classe degli elementi della struttura
- Ogni Visitor concreto
    - Raggruppa i metodi necessari a eseguire una data operazione
    - Nasconde i dettagli di come tale operazione debba essere eseguita


- #text(blue)[*Incapsulamento*]
    - Ogni Visitor deve essere in grado di accedere allo stato degli elementi su cui deve operare
- È difficile aggiungere una nuova classe `ConcreteElement`
    - Per ogni nuova classe `ConcreteElement` è necessario inserire un nuovo metodo `Visit` in tutti i `Visitor` esistenti
        - la gerarchia `Element` deve essere poco o per nulla modificabile - cioè essere #text(blue)[*stabile*]


- #text(blue)[*Visita di elementi non correlati*]
    - Non è necessario che tutti gli elementi da visitare derivino da una classe comune
    `VisitClasseA(ClasseGerarchiaA a);`\
    `VisitClasseB(ClasseGerarchiaB b);`

- #text(blue)[*Stato*]
    - Durante l'operazione ogni Visitor può modificare il proprio stato - ad esempio, per accumulare dei valori o altro



```C
public class CompositeElement : Element

{
    ...
    private List<Element> _children;
    ...
    public override void Accept(Visitor visitor)
    {
        foreach (Element aChild in _children)
            aChild.Accept(visitor);
        visitor.VisitCompositeElement(this);
    }
    ...
}
```


#cfigure("images/2024-05-03-18-29-03.png",90%)


- #text(blue)[*Double dispatch*]
    - L'operazione che deve essere effettuata dipende dal tipo di due oggetti
    - il visitor
    - l'elemento

- `Accept` è un'operazione di tipo #text(blue)[*double dispatch*]

Esempio + EsempioDecorator


==== Anti Pattern

- Oltre ai pattern utili esistono anche gli anti-pattern, che descrivono situazioni ricorrenti e soluzioni notoriamente dannose
- Esempio: _Interface Bloat_, che consiste nell'aggiungere così tante funzionalità a un'interfaccia da renderla impossibile da implementare (o usare!)
- Sostanzialmente, sono soluzioni che non soddisfano i design principle!


==== Pattern ABSTRACT FACTORY

- #text(blue)[*Problema*]:
    - creazione di oggetti connessi o dipendenti tra loro, senza bisogno che il client debba specificare i nomi delle classi concrete all'interno del proprio codice
    - esempio:
```C    
    Menu m;
    if (style == Macos) m = new MacosMenu;
    else if(style == ...) m = new...
```
- la stessa cosa va ripetuta per pulsanti, view...

- #text(blue)[*Requisito*]:
    - si vuole un sistema indipendente da come gli oggetti vengono creati, composti e rappresentati
    - si vuole permettere la configurazione del sistema come scelta tra diverse famiglie di prodotti
    - si vuole che i prodotti che sono organizzati in famiglie siano vincolati ad essere utilizzati con prodotti della stessa famiglia


- #text(blue)[*Soluzione*]:
    - Definizione di una classe
        - astrae la creazione di una famiglia di oggetti
        - istanze diverse costituiscono implementazioni diverse di membri di tale famiglia

#cfigure("images/2024-05-03-18-32-32.png",50%)


- #text(blue)[*Soluzione*]:
    - La creazione dei prodotti è responsabilità delle classi `ConcreteFactory`

#cfigure("images/2024-05-03-18-33-42.png",50%)
#cfigure("images/2024-05-03-18-34-15.png",90%)

- #text(blue)[*Conseguenze*]:
    - isola le classi concrete
        - il codice successivo all'istanziazione è indipendente dalla classe concreta
    - consente di cambiare in modo semplice la famiglia di prodotti utilizzata
        - la coerenza col resto del codice è assicurata dall'utilizzo delle interfacce astratte e non delle classi concrete, secondo l'OCP
    - promuove la coerenza nell'utilizzo dei prodotti


- #text(blue)[*Conseguenze*]:
    - difficile aggiungere supporto per nuove tipologie di prodotti
        - Dato che AbstractFactory definisce tutte le varie tipologie di prodotti che è possibile istanziare, aggiungere una tipologia richiede di modificare l'interfaccia della factory




== Diagramma dei componenti e di Deployment


=== Diagramma dei Componenti

- Da UML 2.0 il concetto di componente si è evoluto
    rispetto alla versione precedente dello standard
- Specifica un #text(blue)[contratto formale di servizi offerti e richiesti]
    in termini di interfacce (eventualmente esposte tramite
    porte)
- Il concetto di componente è strettamente legato a quello
    di #text(blue)[*struttura composita*] che spesso viene impiegata
    per rappresentare le parti interne del componente
- Un componente è tipicamente specificato da uno o più
    classificatori (ad es. classi) e può essere implementato
    da uno o più artefatti (file eseguibile, script, ...)
- Gli internals (parti interne) sono inaccessibili solo
    attraverso le interfacce


==== Struttura Composita

- Il Diagramma di Struttura Composita ha l'obiettivo
    di rappresentare la struttura interna (le parti)
    di un classificatore (classe, componente...),
    inclusi i punti di interazione (porte) utilizzati per accedere
    alle caratteristiche della struttura


#cfigure("images/2024-05-09-10-49-46.png",80%)

- Introdotto per scomporre gerarchicamente
    un classificatore, mostrandone la struttura interna:
       - mostra la struttura interna di un classificatore complesso
       - mostra in modo separato l'#text(blue)[*interfaccia*] di un classificatore
          dalla sua #text(blue)[*struttura interna*]
       - descrive i ruoli che i diversi elementi della struttura
          giocano per soddisfare l'obiettivo della struttura stessa
          e le interazioni richieste
- Questo permette al progettista di prendere un oggetto
    complesso e spezzarlo in parti più piccole e semplici
- È una sorta di strumento di zoom utile per gestire
    la complessità di rappresentazione


==== Package vs. Struttura Composita

- Per capire bene la differenza tra i package e le strutture
    composite bisogna pensare che
       - i primi rappresentano un raggruppamento logico al momento
          dell'analisi
       - mentre le seconde fanno riferimento a quello che succede
          durante l'esecuzione
- Di conseguenza le strutture composite sono adatte
    a rappresentare i componenti e le loro parti,
    e sono usate spesso nei diagrammi dei componenti


#line(length: 100%)

- Da UML 2 è una specializzazione della metaclasse
    `Class`
- Quindi un componente può avere attributi e metodi,
    una struttura interna, porte e connettori
- Da UML 2 l'icona del componente è cambiata

#cfigure("images/2024-05-09-10-51-40.png",90%)

#cfigure("images/2024-05-09-10-52-00.png",90%)
- Interfaccia fornita e interfaccia richiesta devono essere
    compatibili a livello di tipo (attributi e associazioni)
    e di vincoli sul comportamento (operazioni, eventi)

#cfigure("images/2024-05-09-10-52-26.png",60%)

==== Componenti: White-box

#cfigure("images/2024-05-09-10-53-06.png",80%)

- I classificatori interni (internals) che realizzano
    un componente possono essere mostrati in due modi:
       - innestati nel componente

#cfigure("images/2024-05-09-10-53-31.png",60%)


- In modo esplicito tramite la dipendenza di #text(blue)[*realization*]
  #cfigure("images/2024-05-09-10-54-05.png",50%)
- Il componente _Ordine_ è implementato istanziando le classi
    _Ordine_ e _RigaOrdine_
- La #text(blue)[*realization*] è una dipendenza specializzata tra due insiemi
    di elementi di modellazione, di cui uno rappresenta la specifica
    e l'altro una sua implementazione
- Per un componente la #text(blue)[*realization*] definisce i classificatori
    che realizzano il contratto offerto dal componente stesso
    in termini delle sue interfacce offerte e richieste

#cfigure("images/2024-05-09-10-55-01.png",100%)

- Le parti interne sono collegate direttamente tra loro
    (#text(blue)[assembly connector]) oppure connesse a porte
    sul confine del componente (#text(blue)[delegated connector])
- I delegated connector sono utilizzati per esporre servizi
    di una “parte” all'esterno del container


==== Connettori Multiple Wiring
#cfigure("images/2024-05-09-10-56-22.png",90%)
- Entrambi i componenti, Ordine e OrdineInterno,
    richiedono l'interfaccia Persona: l'applicazione non conosce,
    fino al momento dell'esecuzione, quale componente,
    Cliente o Organizzazione, fornirà il servizio richiesto
- Si tratta di #text(blue)[*_un'interazione polimorfica_*]

#cfigure("images/2024-05-09-10-56-41.png",90%)


#line(length: 100%)
- UML 2.0 permette di connettere alla stessa porta
    più interfacce

#cfigure("images/2024-05-09-10-57-23.png",90%)

==== Componenti e sottosistemi

- Mentre in UML 1.x un #text(blue)[*subsystem*] è un tipo di package
- In UML 2 è un tipo di componente
- È quindi possibile specificare per un subsystem
    le interfacce richieste e quelle fornite,
    per evidenziare le relazioni con altri subsystem

#cfigure("images/2024-05-09-10-58-13.png",100%)

#line(length: 100%)

==== Il diagramma

- Il diagramma dei componenti deve essere impiegato
    solamente negli stadi finali della fase di progettazione
    del sistema
- Tale diagramma rispecchia molto da vicino la struttura
    che dovrebbe avere il codice e, in un qualche modo,
    rappresenta l'architettura del sistema
- Potrebbe essere pensato come lo stadio finale
    dell'evoluzione dell'architettura logica
    che in fase di analisi viene rappresentata
    attraverso il diagramma dei package


==== Esempio Villaggio Turistico

#cfigure("images/2024-05-09-10-59-12.png",90%)

=== Diagramma di Deployment

- I diagrammi di deployment #text(blue)[*documentano
    la distribuzione fisica*] di un sistema,
    mostrando i vari pezzi di software in esecuzione
    sulle macchine fisiche
- I diagrammi di deployment quindi mostrano:
    - i collegamenti che permettono la comunicazione fisica
       tra i pezzi hardware
    - le relazioni tra macchine fisiche e processi software,
       con l'indicazione dei vari punti in cui viene eseguito
       il codice


- Gli elementi principali del diagramma:
    - #highlight(fill: myblue)[Artifact]
       - rappresenta una specifica porzione fisica di informazioni utilizzata
          o prodotta dal processo di sviluppo del software
       - esempi di artifact(manufatti): i modelli (un diagramma dei casi d'uso,
          un diagramma delle classi, ...), file sorgenti, script, file eseguibili, ...
       - tipicamente viene utilizzata una relazione di dipendenza
          \<\<manifest>> che illustra gli elementi di modellazione (in genere,
          componenti) utilizzati nella costruzione o generazione di un artefatto
    - #highlight(fill: myblue)[Node]
       - un'unità sulla quali risiedono e/o sono eseguiti componenti/artefatti
       - i nodi comunicano tra loro tramite CommunicationPath
       - l'allocazione degli artefatti su un nodo viene rappresentata
          con una relazione di dipendenza \<\<deploy>> tra il nodo e l'artefatto
    - #highlight(fill: myblue)[Device]
       - è una risorsa fisica computazionale con capacità elaborative
          sulla quale possono essere allocati artefatti per l'esecuzione


- “Manifest” è la relazione di dipendenza che illustra
    gli elementi di modellazione utilizzati nella costruzione
    o generazione di un artefatto
#cfigure("images/2024-05-09-11-01-48.png",90%)

==== Esempio Villaggio Turistico
#cfigure("images/2024-05-09-11-03-02.png",90%)

#line(length: 100%)

- Un #text(blue)[*nodo*] rappresenta qualsiasi cosa possa eseguire
    un lavoro: un server, un device o un'unità organizzativa
- È una risorsa su cui gli artefatti possono essere allocati
    per l'esecuzione, questo fatto viene rappresentato
    con una dipendenza di tipo \<\<deploy>> tra il nodo e l'artefatto
#cfigure("images/2024-05-09-11-03-42.png",80%)
L'immagine ci dice che su ogni `AppServer` viene installato `Order.jar`, e ogni `AppServer` comunica con un `DBServer`.
#cfigure("images/2024-05-09-11-04-33.png",100%)

Tipicamente viene utilizzato il #text(red)[*type-level*], ma l'#text(red)[*instance-level*] è più dettagliato.

#line(length: 100%)

- L'Execution Environment è un nodo che offre l'ambiente per l'esecuzione di specifici tipi di componenti che sono allocati su di esso

```<<OS>>

- <<databasesystem>>
- <<J2EE container>>
- ...
```
#cfigure("images/2024-05-09-11-08-07.png",30%)

Con l'#text(red)[*instance-level*] stiamo specificando che, per far funzionare un particolare componente, su quel dispositivo devono essere presenti tutti quegli artefatti.

- Deployment Specification: è un insieme di proprietà
    che determinano i parametri di esecuzione di un artefatto
    allocato su un nodo
#cfigure("images/2024-05-09-11-10-21.png",90%)

==== Esempio Villaggio Turistico

#cfigure("images/2024-05-09-11-11-01.png",100%)

- In questo caso, `Broker.exe` e `Login.exe` devono essere presenti sulla stessa macchina `ServerLogin`
- C'è un solo `ClientLog`, e possono esserci tanti (indefiniti) `ClientPuntoVendita` e `ClientHotel`
- Nella parte sottostante non è necessario specificare il deployment del componente `ServerLogin`, perché già specificato sopra
- Il fatto che siano separati non implica che ci debbano essere due interfacce di rete diverse, ma solo che i percorsi sono diversi

*Importante:* in una applicazione web il deployment degli artefatti che manifestano #underline[un client] non avviene sul client in sè, ma sul server; il client in questi casi è un semplice browser web.



== Progettazione Concettuale (E/R)

#cfigure("images/2024-05-09-16-52-44.png",90%)

=== Raccolta dei requisiti

- I requisiti devono innanzitutto essere acquisiti
- Le fonti possono essere molto diversificate tra loro:
    - #text(teal)[utenti], attraverso:
       - interviste
       - documentazione apposita
    - #text(teal)[documentazione esistente]:
       - normative (leggi, regolamenti di settore)
       - regolamenti interni, procedure aziendali
       - realizzazioni preesistenti
    - #text(teal)[modulistica]
- La raccolta dei requisiti è un'attività difficile
    e non standardizzabile
       - in genere procede di pari passo con la fase di analisi
          (la prima analisi stimola nuove domande, ecc...)


=== Interagire con gli utenti

- È un'attività da considerare con molta attenzione,
    in quanto:
       - utenti diversi possono fornire informazioni diverse
       - utenti a livello più alto hanno spesso una visione più ampia
          ma meno dettagliata
- In generale, risulta utile:
    - effettuare spesso #text(teal)[verifiche] di comprensione e coerenza
    - verificare anche per mezzo di #text(teal)[esempi]
       (generali e relativi a casi limite)
    - richiedere #text(teal)[definizioni] e #text(teal)[classificazioni]
    - far evidenziare gli #text(teal)[aspetti essenziali] rispetto a quelli marginali


=== Requisiti: documentazione descrittiva

- Regole generali:
    - scegliere il corretto #text(teal)[livello di astrazione]
    - #text(teal)[standardizzare] la struttura delle frasi
    - #text(teal)[suddividere] le frasi articolate
    - #text(teal)[separare] le frasi sui #text(teal)[dati] da quelle sulle #text(teal)[funzioni]
       (operazioni)
- Per meglio evidenziare i concetti che sono espressi nei requisiti, è opportuno:

    - costruire un #text(teal)[glossario dei termini]
    - individuare #text(green)[omonimi] e #text(green)[sinonimi] e unificare i termini
    - rendere esplicito il #text(green)[riferimento fra termini]
    - riorganizzare le frasi per concetti


==== Esempio: BD bibliografica (1)


Si vogliono organizzare i dati di interesse per
automatizzare la gestione dei riferimenti bibliografici,
con tutte le informazioni da riportarsi in una bibliografia.

Per ogni pubblicazione deve esistere
un codice identificante #underline[costituito da sette caratteri,
indicanti le iniziali degli autori, l'anno di pubblicazione
e un carattere aggiuntivo per la discriminazione
delle collisioni] (ad es. BL2007a)

- Dettagli marginali tendono solo a distrarre
    e non forniscono nessuna indicazione sulla struttura
    dello schema che si deve progettare

==== Esempio: BD bibliografica (2)

Si vogliono organizzare i dati di interesse per
automatizzare la gestione dei riferimenti bibliografici,
con tutte le informazioni da riportarsi in una bibliografia.

*Le pubblicazioni sono di due tipi, monografie (per le quali
interessano editore, data e luogo di pubblicazione) e articoli su
rivista (con nome della rivista, volume, numero, pagine
e anno di pubblicazione); per entrambi i tipi
si debbono ovviamente riportare i nomi degli autori.*

Per ogni pubblicazione deve esistere un codice
identificante...

- Il paragrafo in grassetto fornisce informazioni utili per derivare
    lo schema concettuale, in quanto introduce concetti importanti
    nella realtà in esame


==== Un altro esempio più articolato

- Si vuole realizzare una base di dati per una società che eroga corsi,
    di cui vogliamo rappresentare i dati deipartecipantiai corsi e deidocenti.
- Per gli studenti(circa 5000), identificati da un codice, si vuole memorizzare
    il codice fiscale, il cognome, l'età, il sesso, il luogodi nascita, il nome dei loro
    attuali datori di lavoro, i posti dove hanno lavorato in precedenza insieme al periodo,
    l'indirizzo e il numero di telefono, icorsiche hanno frequentato
    (i corsi sono in tutto circa 200) e il giudizio finale.
- Rappresentiamo anche iseminariche stanno attualmente frequentando e,
    per ogni giorno, i luoghie le ore dove sono tenute le lezioni.
- I corsi hanno un codice, untitoloe possono avere varie edizioni con date di inizio
    e fine e numero di partecipanti.
- Se gli studenti sono liberi professionisti, vogliamo conoscere l'area di interesse e,
    se lo possiedono, iltitolo. Per quelli che lavorano alle dipendenze di altri,
    vogliamo conoscere invece il loro livello e la posizione ricoperta.
- Per gliinsegnanti(circa 300), rappresentiamo il cognome, l'età, ilpostodove sono
    nati, il nome del corso che insegnano, quelli che hanno insegnato nel passato
    e quelli che possono insegnare. Rappresentiamo anche tutti i loro recapiti telefonici.
    I docenti possono essere dipendenti interni della società o collaboratori esterni.


==== Glossario dei termini, omonimi e sinonimi

- Raramente i requisisti espressi in linguaggio naturale
    sono privi di ambiguità. È infatti frequente il caso di\
       #text(teal)[Omonimi]: lo stesso termine viene usato per descrivere concetti
          differenti
          (es: libro e copia di libro, posto: di lavoro e geografico)\
       #text(teal)[Sinonimi]: termini diversi vengono usati per descrivere lo stesso
          concetto
          (es: studente e partecipante)
- Un modo conveniente per rappresentare sinteticamente
    i concetti più rilevanti emersi dall'analisi è il glossario dei
    #text(teal)[termini], il cui scopo è fornire per ogni concetto rilevante:
       - Una breve descrizione del concetto
       - Eventuali sinonimi
       - Relazioni con altri concetti del glossario stesso


===== Esempio

#cfigure("images/2024-05-09-17-00-53.png",90%)

==== Ristrutturazione dei requisiti

- Oltre a costruire il glossario, al fine di semplificare
    le analisi successive, è utile riformulare i requisiti:
       - Eliminare le omonimie
       - Usare un termine univoco per ogni concetto
       - Riorganizzare le frasi raggruppandole in base al concetto
          cui si riferiscono
          Nell'esempio:
             - Frasi di carattere generale
             - Frasi riferite ai partecipanti
             - Frasi riferite ai docenti
             - Frasi riferite ai corsi
             - Frasi riferite alle società


===== Esempio: frasi relative ai partecipanti

Per i partecipanti (circa 5000), identificati
da un codice, rappresentiamo il codice fiscale,
il cognome, l'età, il sesso, la città di nascita,
i nomi dei loro attuali datori di lavoro
e di quelli precedenti (insieme alle date di inizio
e fine rapporto), le edizioni dei corsi che stanno
attualmente frequentando e quelli che hanno
frequentato nel passato, con la relativa
votazione finale.


==== Dai concetti allo schema E/R

- Va sempre ricordato che un concetto non è di per sé
    un'entità, un'associazione, un attributo, o altro
       #text(teal)[DIPENDE DAL CONTESTO!]
- Come regole guida, un concetto verrà rappresentato
    come
       - Entità
          - se ha proprietà significative e descrive oggetti con esistenza
             autonoma
       - Attributo
          - se è semplice e non ha proprietà
       - Associazione
          - se correla due o più concetti
       - Generalizzazione/specializzazione
          - se è caso più generale/particolare di un altro


==== Strategie di progettazione

- Per affrontare progetti complessi è opportuno adottare
    uno specifico modo di procedere, ovvero una #text(green)[strategia di
    progettazione]
- I casi notevoli sono:
    - Strategia #highlight(fill: teal)[top-down:]\
       Si parte da uno schema iniziale molto astratto ma completo,
       che viene successivamente raffinato fino ad arrivare
       allo schema finale
    - Strategia #highlight(fill: teal)[bottom-up:]\
       Si suddividono le specifiche in modo da sviluppare semplici
       schemi parziali ma dettagliati, che poi vengono integrati tra loro
    - Strategia #highlight(fill: teal)[inside-out:]\
       Lo schema si sviluppa “a macchia d'olio”, partendo dai concetti
       più importanti, che quindi vengono espansi aggiungendo
       quelli a essi correlati, e così via


===== Pro e controlo delle strategie

#cfigure("images/2024-05-09-17-08-07.png",90%)

===== Un approccio “misto”

- Nella pratica si fa spesso uso di una strategia ibrida,
    nella quale:

    + si individuano i concetti principali e si realizza uno #text(teal)[schema scheletro], che contiene solamente i concetti più importanti
    + sulla base di questo si può #text(teal)[decomporre]
    + poi si raffina, si espande, si integra

... vediamo cosa succede nel caso della società di
formazione...


===== Società di formazione: schema scheletro

#cfigure("images/2024-05-09-17-10-42.png",70%)

#heading(level: 5, numbering: none)[Raffinamento di Partecipanti]

#cfigure("images/2024-05-09-17-11-51.png",90%)
- Dato che i partecipanti possono essere o dipendenti o professionisti, facciamo una gerarchie totale ed esclusiva (t,e)
- I dipendenti possono avere (impiego corrente) un unico datore di lavoro, mentre i datori possono avere da 0 a n dipendenti
- Impiego passato si riferisce alla classe generale Partecipanti perché uno che adesso è un Professionista può essere stato, in passato, un Dipendente
#heading(level: 5, numbering: none)[Raffinamento di Corsi]

#cfigure("images/2024-05-09-17-12-33.png",90%)
- La chiave di Edizioni Corsi è il Codice del corso e la Data inizio
#heading(level: 5, numbering: none)[Raffinamento di Docenti]

#cfigure("images/2024-05-09-17-13-09.png",90%)

#heading(level: 5, numbering: none)[Integrazione: schema di riferimento]

#cfigure("images/2024-05-09-17-14-14.png",90%)

#heading(level: 5, numbering: none)[Integrazione: Partecipanti e Corsi]
#cfigure("images/2024-05-09-17-17-59.png",90%)

#heading(level: 5, numbering: none)[Integrazione: Docenti e Corsi]

#cfigure("images/2024-05-09-17-18-50.png",90%)
- C'è un vincolo non espresso sulla data di Docenza corrente e Docenza passata
==== Qualità di uno schema concettuale

- #text(teal)[Lo schema E/R deve essere verificato] accuratamente per verificare che risponda a requisiti di:

- #highlight(fill: teal)[Correttezza]
    - Non devono essere presenti errori (sintattici o semantici)
- #highlight(fill: teal)[Completezza]
    - Tutti i dati di interesse devono essere specificati
- #highlight(fill: teal)[Leggibilità]
    - Riguarda anche aspetti prettamente estetici dello schema
- #highlight(fill: teal)[Minimalità]
    - È importante capire se esistono elementi ridondanti
       nello schema; in alcuni casi ciò non è un problema,
       ma può essere viceversa una scelta di progettazione
       volta a favorire l'esecuzione di certe operazioni


==== Metodologia basata sulla strategia mista

#text(green)[Analisi dei requisiti]
- Analizzare i requisiti ed eliminare le ambiguità
- Costruire un glossario dei termini, raggruppare i requisiti
#text(green)[Passo base]
- Definire uno schema scheletro con i concetti più rilevanti
#text(green)[Passo di decomposizione] (se necessario o appropriato)
- Decomporre i requisiti con riferimento ai concetti nello schema scheletro
#text(green)[Passo iterativo] (da ripetere finché non si è soddisfatti)
- Raffinare i concetti presenti sulla base delle loro specifiche
- Aggiungere concetti per descrivere specifiche non descritte
#text(green)[Passo di integrazione] (se si è decomposto)
- Integrare i vari sottoschemi in uno schema complessivo, facendo riferimento allo schema scheletro
#text(green)[Analisi di qualità] (ripetuta e distribuita)
- Verificare le qualità dello schema e modificarlo


==== Riassumendo

- La #text(teal)[fase di analisi dei requisiti] è fondamentale
    per poter progettare una base di dati che rispetti
    i requisiti
- Mancando la possibilità di standardizzarla,
    tale fase si avvale necessariamente di regole
    di buon senso e di una serie di strumenti che riducono
    il rischio di commettere errori grossolani,
    oltre a costituire una valida documentazione
- Per la #text(green)[progettazione dello schema E/R] sono possibili
    diverse strategie, di cui quella #text(teal)[mista] è senz'altro
    la più diffusa e adeguata anche nel caso
    di progetti estremamente complessi



== Dalla Progettazione all'Implementazione


=== Introduzione

- Arrivati a questo punto apparentemente
    occorre “solamente” implementare tutte le classi
    che abbiamo individuato nelle varie fasi
    (analisi, progettazione, ecc.)
- “Sfortunatamente” anche a questo livello
    occorre effettuare delle scelte progettuali
    che hanno un impatto sulle caratteristiche del SW
    (efficienza, riusabilità, ecc.)
- Il #text(blue)[*progetto di dettaglio*] rappresenta
    una descrizione del sistema molto vicina alla codifica,
    ovvero che la vincola in maniera sostanziale
       - Per esempio, descrivendo non solo le classi in astratto
          ma anche i loro attributi e metodi, con relativi tipi e firma


=== Progettazione di Dettaglio

- Durante la progettazione di dettaglio
    è necessario definire
       - #highlight(fill: myblue)[Tipi di dato]
          che non sono stati definiti nel modello OOA
       - #highlight(fill: myblue)[Navigabilità delle associazioni] tra classi
          e relativa implementazione
       - #highlight(fill: myblue)[Strutture dati]
          necessarie per l'implementazione del sistema
       - #highlight(fill: myblue)[Operazioni]
          necessarie per l'implementazione del sistema
       - #highlight(fill: myblue)[Algoritmi]
          che implementano le operazioni
       - #highlight(fill: myblue)[Visibilità] di classi, (attributi,) operazioni, ...


==== Navigabilità di un'Associazione

- Possibilità di spostarsi da un qualsiasi oggetto
    della classe origine a uno o più oggetti
    della classe destinazione (a seconda della molteplicità)
- I messaggi possono essere inviati
    solo nella direzione della freccia Ogni docente deve avere un riferimento al proprio
       dipartimento di afferenza


#cfigure("images/2024-05-17-17-38-14.png",90%)



- #text(blue)[*A livello di analisi del problema*],
    le associazioni di composizione e di aggregazione
    hanno una direzione precisa detti #text(blue)[*A il contenitore*]
    e #text(blue)[*B l'oggetto contenuto*], è A che contiene B,
    e non viceversa
- #text(blue)[*A livello implementativo*], un'associazione può essere
    - #text(blue)[*mono-direzionale*] quando
       da A si deve poter accedere a B, ma non viceversa
    - #text(blue)[*bi-direzionale*] quando
       da A si deve poter accedere a B e
       da B si deve poter accedere velocemente ad A


- Dal punto di vista implementativo, la #text(blue)[*bi-direzionalità*]
    - è molto efficiente
    - ma occorre tenere sotto controllo la consistenzadelle strutture
       dati utilizzate per la sua implementazione

#cfigure("images/2024-05-17-17-39-28.png",90%)

==== Implementazione delle Associazioni

#heading(level: 5, numbering: none)[Associazioni con molteplicità 0..1 o 1..1]
- Aggiungere alla classe cliente un attributo membro
    che rappresenta
       - il #text(blue)[*riferimento all'oggetto*] della classe fornitore
       - e/o l'#text(blue)[*identificatore univoco*] dell'oggetto della classe
          fornitore (solo se persistente)
       - o il #text(blue)[*valore dell'oggetto*] della classe fornitore
          (solo nel caso di #text(blue)[*composizione*] e #text(blue)[*molteplicità 1..1*])

#cfigure("images/2024-05-17-17-39-50.png",60%)
#cfigure("images/2024-05-17-17-40-07.png",60%)

#heading(level: 5, numbering: none)[Associazioni con molteplicità 0..\* o 1..\*]
- Aggiungere alla classe cliente un attributo membro
    che referenzia un'#text(blue)[*istanza di una classe contenitore*]
- Una #text(blue)[*classe contenitore*] è una classe le cui istanze
    sono #text(blue)[*collezioni*] di (riferimenti a) oggetti della classe
    fornitore
- La classe contenitore può essere
    - realizzata, oppure
    - presa da una libreria (preferibilmente) 


#cfigure("images/2024-05-17-17-40-36.png",80%)
#cfigure("images/2024-05-17-17-40-51.png",80%)

#cfigure("images/2024-05-17-17-41-03.png",80%)
#cfigure("images/2024-05-17-17-41-12.png",80%)
#cfigure("images/2024-05-17-17-41-22.png",80%)


==== Classi Contenitore

- Una #text(blue)[*classe contenitore*] (o semplicemente contenitore)
    è una classe le cui istanze contengono oggetti
    di altre classi
- Se gli oggetti contenuti sono in #text(blue)[*numero fisso*],
    è sufficiente un vettore predefinito del linguaggio
- Se gli oggetti contenuti sono in #text(blue)[*numero variabile*],
    un vettore predefinito non basta e occorre
    una classe contenitore
- Esempi di classi contenitore sono
    - Vettori, stack, liste, alberi, ...
- Funzionalità minime di una classe contenitore
    - #text(blue)[*Inserire, rimuovere, trovare*] un oggetto in una collezione
    - #text(blue)[*Enumerare*] (iterare su) gli oggetti della collezione


- I contenitori possono essere classificati in funzione
    - del modo in cui contengono gli oggetti
       - #text(blue)[*contenimento per riferimento*]: gli oggetti sono reference type
       - #text(blue)[*contenimento per valore*]: gli oggetti sono value type
    - dell'omogeneità o eterogeneità di tali oggetti
       - #text(blue)[*oggetti omogenei*]: tutti gli oggetti contenuti sono dello stesso tipo
       - #text(blue)[*oggetti eterogenei*]: gli oggetti contenuti possono essere di tipo diverso


===== Contenimento per Riferimento

- L'oggetto contenuto #text(blue)[*esiste per conto proprio*]
- L'oggetto contenuto può essere #text(blue)[*in più contenitori
    contemporaneamente*]
- Quando un oggetto viene inserito in un contenitore,
    #text(blue)[*non viene duplicato*]
    ma ne viene memorizzato solo il riferimento
- La distruzione del contenitore non comporta
    la #text(blue)[*distruzione degli oggetti contenuti*]


===== Contenimento per Valore

- L'oggetto contenuto
    - viene memorizzato nella struttura dati del contenitore
    - esiste solo in quanto contenuto fisicamente
       in un altro oggetto
- Quando un oggetto deve essere inserito
    in un contenitore, #text(blue)[*viene duplicato*]
- La distruzione del contenitore comporta
    la #text(blue)[*distruzione degli oggetti contenuti*]


===== Contenimento di Oggetti Omogenei

- Per implementare #text(blue)[*contenitori di oggetti omogenei*]
    (sia per valore, sia per riferimento)
    sono ideali le #text(blue)[*classi generiche*]
- Il tipo degli oggetti contenuti viene lasciato generico
    e ci si concentra sugli algoritmi di gestione della
    collezione di oggetti
- Quando serve una classe contenitore di oggetti
    appartenenti a una classe specifica,
    #text(blue)[*è sufficiente istanziare la classe generica*],
    specificando il tipo desiderato


#cfigure("images/2024-05-17-17-42-03.png",60%)
#cfigure("images/2024-05-17-17-42-12.png",80%)


===== Contenimento di Oggetti Eterogenei

- Per implementare #text(blue)[*contenitori di oggetti eterogenei*]
    (solo per riferimento) è necessario usare l'#text(blue)[*ereditarietà*]
    e sfruttare la proprietà che un puntatore
    alla superclasse radice della gerarchia può puntare
    a un'istanza di una qualunque sottoclasse
- La classe contenitore può essere generica,
    ma il tipo deve essere la #text(blue)[*superclasse radice
    della gerarchia*] (nel peggiore dei casi, `object`)


=== Implementazione delle Associazioni

- Un modo alternativo per implementare un'associazione
    tra due oggetti è tramite un #text(blue)[*dizionario*]
- Un dizionario è un tipo particolare di contenitore,
    che associa due oggetti: #text(blue)[*la chiave e il rispettivo valore*]
- La chiave
    - Può essere un oggetto qualsiasi, non necessariamente una stringa o un intero
    - Deve essere unica
- Il dizionario, data una chiave, ritrova in modo efficiente
    il valore ad essa associato


#cfigure("images/2024-05-17-17-42-33.png",85%)

==== Identificazione degli Oggetti

- Un oggetto (contenitore o meno) può contenere un #text(blue)[*riferimento univoco*] a un altro oggetto
- Come è possibile #text(blue)[*identificare univocamente*] un oggetto per poterlo associare a un altro?
- Nel caso di #text(blue)[*strutture dati interamente contenute nello spazio di indirizzamento*] dell'applicazione, un oggetto può essere identificato univocamente mediante il suo #text(blue)[*indirizzo*] (logico) #text(blue)[*di memoria*]


- Nel caso di #text(blue)[*database*] o di #text(blue)[*sistemi distribuiti*],
    a ogni oggetto deve essere associato
    un #text(blue)[*identificatore univoco persistente*] tramite il quale
    deve essere possibile risalire all'oggetto stesso,
    sia che risieda in memoria, su disco o in rete
- L'identificatore univoco è un attributo che al momento
    della creazione dell'oggetto viene inizializzato con:
       - un valore generato automaticamente dal sistema
       - il valore della chiave primaria di una tabella relazionale, ...
- Il nome di tale attributo #underline[*potrebbe*] essere
    - idDocente
    - idStudente, ...


===== Un Esempio Reale

- La #text(blue)[*tecnologia COM*] (MS) permette a un'applicazione
di trovare, caricare e utilizzare _run-time_ i #text(blue)[*componenti*]
    necessari per la sua esecuzione
- Ogni componente è memorizzato in una DLL
    (#text(red)[Dynamic Link Library]) - un file locale o remoto
- Quando l'applicazione ha bisogno di un componente,
    #text(blue)[*il sistema deve essere in grado di localizzare la DLL*]
    che contiene quel particolare componente



- L'indipendenza dalla collocazione fisica
    non consente di utilizzare un indirizzo fisico (_pathname_)
- Pertanto, deve essere utilizzato un #text(red)[meccanismo
    di indirizzamento logico] che permetta di identificare
    univocamente il file che contiene il componente
- Si utilizzano degli #text(red)[identificatori globali]
    (#text(blue)[*GUID*] = _Globally Unique Identifier_)




- Il concetto di GUID è stato introdotto,
    con un nome leggermente diverso (#text(blue)[UUID] = Universally
    Unique Identifier), dall'OSF (Open Software Foundation)
    nelle specifiche #text(blue)[DCE] (Distributed Computing
    Environment)
- In DCE gli UUID vengono utilizzati per identificare
    i destinatari delle chiamate di procedura remota (RPC)




- Un GUID è un numero di 128 bit (16 byte)
    generato in modo da garantire l'unicità nello spazio
    e nel tempo: MAC (48/64 bit) + ticks (64 bit - 100ns)
    rappresentato così:\
    `{32bb8320-b41b-11cf-a6bb-0080c7b2d682}`
- COM utilizza diversi tipi di GUID
- Il tipo più importante di GUID serve a identificare
    le classi di componenti:
    ogni classe di componenti COM è caratterizzata
    da un proprio identificatore che viene chiamato
    #text(blue)[*CLSID*] (Class Identifier)




- Disponendo di un CLSID, un'applicazione può chiedere
    alla funzione di sistema #text(blue)[`CoCreateInstance`]
    di creare un istanza del componente e di restituire
    un riferimento nel spazio di indirizzamento
    dell'applicazione stessa
- Il database di sistema di Windows (#text(blue)[*registry*])
    mantiene una corrispondenza tra CLSID ed entità fisiche
    (DLL, EXE) che contengono l'implementazione
    dei componenti (server)



- `CoCreateInstance` provvede a
    - reperire il server tramite il _registry_
    - caricarlo in memoria (se non è già presente)
    - creare un'istanza e restituirne un riferimento
#cfigure("images/2024-05-17-17-43-16.png",90%)


- In .NET esiste la classe `System.Guid` che permette di gestire istanze di GUID

- Ad esempio, per ottenere un nuovo GUID, è sufficiente
    invocare il metodo statico
    `Guid.NewGuid()` che, ovviamente, restituisce un `System.Guid`
- Altri metodi e operatori permettono di confrontare GUID



===== Modifiche per Utilizzare il Livello di Ereditarietà Supportato

- Se esistono strutture con ereditarietà multipla
- Se il linguaggio di programmazione non ammette
    l'ereditarietà multipla
- È necessario convertire
    le strutture con ereditarietà multipla
    in strutture con solo ereditarietà semplice



#cfigure("images/2024-05-17-17-43-39.png",70%)

#heading(level: 6, numbering: none)[$1^a$ possibilità (composizione e delega)]
- Scegliere la più significativa tra le superclassi
    ed ereditare esclusivamente da questa
- Tutte le altre superclassi diventano possibili “ruoli”
    e vengono connesse mediante composizione
- Le caratteristiche delle superclassi escluse
    vengono incorporate nella classe specializzata
    tramite composizione e delega
    e non tramite ereditarietà
#cfigure("images/2024-05-17-17-43-54.png",65%)




#heading(level: 6, numbering: none)[$2^a$ possibilità (interfaccia)]
- Appiattire tutto in una gerarchia semplice
    e implementare un'interfaccia
- In questo modo, una o più relazioni di ereditarietà
    si perdono e #text(blue)[*gli attributi e le operazioni corrispondenti
    devono essere ripetuti nelle classi specializzate*]



#cfigure("images/2024-05-17-17-44-15.png",65%)
#cfigure("images/2024-05-17-17-44-26.png",68%)

=== Miglioramento delle Prestazioni

- Il software con le prestazioni migliori
    - fa la cosa giusta “abbastanza velocemente”
       (cioè, soddisfacendo i requisiti e/o le attese
       del cliente)
    - pur rimanendo entro costi e tempi preventivati
- Per migliorare la velocità percepita può bastare
    - la #text(blue)[*memorizzazione di risultati intermedi*]
    - un'#text(blue)[*accurata progettazione dell'interazione con l'utente*]
       (ad es. utilizzando multi-threading)
- Un #text(blue)[*traffico di messaggi molto elevato tra oggetti*]
    può invece richiedere dei cambiamenti
    per aumentare la velocità


- Di norma, la soluzione è che un oggetto possa accedere
    direttamente ai valori di un altro oggetto (aggirando
    l'incapsulamento!)
       - Utilizzare metodi inline
       - Utilizzare la dichiarazione friend
       - Combinare insieme due o più classi
- Questo tipo di modifica deve essere presa
    in considerazione solo dopo che tutti gli altri aspetti
    del progetto sono stati soggetti a misure e modifiche
- L'unico modo per sapere se una modifica contribuirà
    in modo significativo a rendere il software “abbastanza
    veloce” è tramite le misure e l'osservazione








= Framework .NET

== Introduzione

=== Tecnologia COM - Component Object Model

- Nasce nel 1993
- COM è un sistema platform-independent, distribuito,
    object-oriented per la creazione di componenti
    software binari
- COM non è un linguaggio object-oriented
    ma uno #text(blue)[*standard*]
- COM specifica un modello a oggetti
    e requisiti di programmazione che permettono
    agli oggetti COM di interagire con altri oggetti

- Ad esempio un programma Java può parlare con un programma C, utilizzando la Java Native Interface

- COM fa la stessa cosa del linker C quando abbiamo bisogno che più programmi parlino tra loro (link dinamico)

Esempio di oggetti COM che parlano tra di loro:

```C
interface IUnknown
{
    virtual HRESULT QueryInterface(IID iid, void *ppvObject) = 0;
    virtual ULONG AddRef(void) = 0;
    virtual ULONG Release(void) = 0;
};
```

- `QueryInterface` è utilizzata per ottenere un puntatore
    a un'altra interfaccia (a tempo di esecuzione, non di compilazione), dato un `GUID` che identifica univocamente
    tale interfaccia (noto comunemente come interfaceID, o IID)
       - se l'oggetto COM non implementa tale interfaccia,
          viene restituito un errore `E_NOINTERFACE`
- `AddRef` è utilizzato dai client per indicare che un oggetto COM
    viene referenziato (usato)
- `Release` è utilizzato dai client per indicare che hanno finito
    di utilizzare l'oggetto COM



- Le specifiche COM richiedono l'utilizzo di una tecnica
    chiamata #text(blue)[*reference counting*] per assicurare
    che i singoli oggetti rimangano “vivi” fintantoché esistono
    client che hanno ottenuto l'accesso
    a una o più delle loro interfacce e, di converso,
    che gli stessi oggetti vengano appropriatamente
    cancellati quando i clienti che li utilizzavano
    hanno finito di usarli e non ne hanno più bisogno; il contatore viene incrementato a ogni `AddRef` e decrementato a ogni `Release`

- #text(blue)[Un oggetto COM è responsabile della liberazione della propria memoria] una volta che il suo reference count arrivi a zero

- Il reference counting può causare #text(blue)[problemi se due o più oggetti hanno riferimenti circolari]




- Ereditarietà solo con composizione e delega

#cfigure("images/2024-04-11-09-22-06.png",80%)


- La #text(blue)[*posizione*] di ciascun componente è memorizzata
    nel registro Windows

- Di un certo componente può esistere un'unica versione installata

- Questa limitazione può complicare seriamente
il deployment di applicazioni basate su COM,
a causa della possibilità che diversi programmi,
o anche diverse versioni dello stesso programma,
siano progettati per funzionare con versioni diverse
dello stesso componente COM

- Questa situazione è nota anche come inferno delle DLL (DLL hell)


=== Cos'è il Framework .NET

- Ambiente di esecuzione (runtime environment) + Libreria di classi (standard + estensioni MS)
- Versione 1.0 del 2002 ► v. 4.8 (versione definitiva, 7/19)
- Semplifica lo sviluppo e il deployment
- Aumenta l'affidabilità del codice
- Unifica il modello di programmazione
- È completamente indipendente da COM
- È fortemente integrato con COM


- Ambiente object-oriented
    - Qualsiasi entità è un oggetto
    - Classi ed ereditarietà pienamente supportati
- Riduzione errori comuni di programmazione
    - #text(blue)[*Garbage Collector*]
    - Linguaggi fortemente tipizzati - #text(blue)[*Type Checker*]
    - Errori non gestiti ► generazione di eccezioni



- Libertà di scelta del linguaggio
    - Funzionalità del framework disponibili in tutti i linguaggi
       .NET
    - I componenti della stessa applicazione
       possono essere scritti in linguaggi diversi
    - Ereditarietà supportata anche tra linguaggi diversi
- Possibilità di estendere una qualsiasi classe .NET
    (non sealed) mediante ereditarietà
- Diversamente da COM:
    - si usa e si estende la classe stessa
    - non si deve utilizzare composizione e delega


- .NET è un'implementazione di CLI
    - Common Language Infrastructure
- CLI e il linguaggio C\# sono standard ECMA
    - ECMA-334 (C\#), ECMA-335 (CLI)
- Esistono altre implementazioni di CLI:
    - SSCLI (Shared Source CLI by Microsoft, per Windows,
       FreeBSD e Macintosh) - Rotor
    - Mono (per Linux)
    - DotGNU
    - Intel OCL (Open CLI Library)
    - ...


=== Standard ECMA-335

- Definisce la Common Language Infrastructure (CLI)
    nella quale applicazioni scritte in #text(blue)[*diversi linguaggi
    di alto livello*] possono essere eseguite in #text(blue)[*diversi
    ambienti di sistema*] senza la necessità di riscrivere
    l'applicazione per prendere in considerazione
    le caratteristiche peculiari di tali ambienti
- CLI è un #text(blue)[*ambiente a tempo di esecuzione*], con:
    - un formato di file
    - un sistema di tipi comune
    - un sistema di metadati estensibile
    - un linguaggio intermedio
    - accesso alla piattaforma sottostante
    - una libreria di classi base


- Concetti chiave:
    - #highlight(fill: myblue)[(Microsoft) Intermediate Language] - (MS)IL
    - #highlight(fill: myblue)[Common Language Runtime] - CLR
       - ambiente di esecuzione runtime per le applicazioni .NET
       - il codice che viene eseguito sotto il suo controllo
          si dice *codice gestito* (managed)
    - #highlight(fill: myblue)[Common Type System] - CTS
       - tipi di dato supportati dal framework .NET
       - consente di fornire un modello di programmazione unificato
    - #highlight(fill: myblue)[Common Language Specification] - CLS
       - regole che i linguaggi di programmazione devono seguire
          per essere interoperabili all'interno del framework .NET
       - sottoinsieme di CTS


=== Codice interpretato

#cfigure("images/2024-04-11-09-29-02.png",90%)


=== Codice nativo

#cfigure("images/2024-04-11-09-29-30.png",90%)

=== Codice IL

#cfigure("images/2024-04-11-09-31-18.png",90%)
JIT: just in time
#cfigure("images/2024-04-11-09-33-15.png",90%)

#cfigure("images/2024-04-11-09-32-32.png",90%)


=== Assembly

- Unità minima per la distribuzione e il versioning
- Normalmente è composto da un solo file

#cfigure("images/2024-04-11-09-33-50.png",90%)


- Ma può essere composto anche da più file

#cfigure("images/2024-04-11-09-35-05.png",70%)

=== Metadati

- Descrizione dell'assembly - Manifest
    - Identità: nome, versione, cultura [, public key]
    - Lista dei file che compongono l'assembly
    - Riferimenti ad altri assembly da cui si dipende
    - Permessi necessari per l'esecuzione
    - ...
- Descrizione dei tipi contenuti nell'assembly
    - Nome, visibilità, classe base, interfacce
    - Campi (attributi membro), metodi, proprietà, eventi, ...
    - Attributi (caratteristiche aggiuntive)
       - definiti dal compilatore
       - definiti dal framework
       - definiti dall'utente


=== Chi usa i metadati?

- Compilatori
    - Compilazione condizionale
- Ambienti RAD (Rapid Application Development)
    - Informazioni sulle proprietà dei componenti
       - Categoria
       - Descrizione
       - Editor specializzati per tipo di proprietà
- Tool di analisi dei tipi e del codice
    - Intellisense, ILDASM, Reflector, ...
- Sviluppatori - *Reflection* (introspezione)
    - Analisi del contenuto di un assembly, permetter agli altri di guardare dentro


=== Esempio Assembly

```yasm
.assembly Hello { }
.assembly extern mscorlib { }
.method public static void main()
{
    .entrypoint
    ldstr "Hello IL World!"
    call void [mscorlib]System.Console::WriteLine
    (class System.String)
    ret
}

ilasm helloil.il
```

=== Dove trovare gli Assembly

- #highlight(fill: myblue)[Assembly privati]
    - Utilizzati da un'applicazione specifica
    - Directory applicazione (e sub-directory)
- #highlight(fill: myblue)[Assembly condivisi]
    - Utilizzati da più applicazioni
    - Global Assembly Cache (GAC)
    - `c:\windows\assembly`
- #highlight(fill: myblue)[Assembly scaricati da URL]
    - Download cache
    - `c:\windows\assembly\download`


=== Deployment semplificato

- Installazione senza effetti collaterali
    - Applicazioni e componenti possono essere
       - condivisi 
       - privati
- Esecuzione side-by-side
    - Versioni diverse dello stesso componente
       possono coesistere, anche nello stesso processo


=== Common Language Runtime

#cfigure("images/2024-04-11-09-38-34.png",60%)


- IL CLR offre vari servizi alle applicazioni

#cfigure("images/2024-04-11-09-39-17.png",75%)

#cfigure("images/2024-04-11-09-40-06.png",70%)


=== Garbage Collector

- Gestisce il ciclo di vita di tutti gli oggetti .NET
- Gli oggetti vengono distrutti automaticamente
    quando non sono più referenziati
- A differenza di COM,
    non si basa sul Reference Counting
       - #underline(stroke: 1.5pt + green)[Maggiore velocità di allocazione] 
       - #underline(stroke: 1.5pt + green)[Consentiti i riferimenti circolari] 
       - #underline(stroke: 1.5pt + red)[Perdita della distruzione deterministica]: non posso sapere in che momento un determinato oggetto verrà distrutto 
       - Inoltre, essendo più complicato del garbage collector di COM, l'esecuzione dello stesso è un po' più pesante


=== Gestione delle eccezioni

- Praticamente uguali a quelle di Java

- Un'eccezione è
    - una condizione di errore
    - un comportamento inaspettato
incontrato durante l'esecuzione del programma
- Un'eccezione può essere generata da
    - codice del programma in esecuzione
    - ambiente di runtime
- In CLR, un'eccezione è un oggetto che eredita
    dalla classe `System.Exception`
- Gestione uniforme, elimina
    - codici `HRESULT` di COM
    - codici di errore Win32
    - ...



- Concetti universali
    - Lanciare un'eccezione (`throw`)
    - Catturare un'eccezione (`catch`)
    - Eseguire codice di uscita da un blocco controllato
       (`finally`)
- Disponibile in tutti i linguaggi .NET con sintassi diverse


=== Common Type System

- Tipi di dato supportati dal framework .NET
    - Alla base di tutti i linguaggi .NET
- Fornisce un modello di programmazione unificato
- Progettato per linguaggi object-oriented, procedurali
    e funzionali
       - Esaminate caratteristiche di 20 linguaggi
       - Tutte le funzionalità disponibili con IL
       - Ogni linguaggio utilizza alcune caratteristiche


- Alla base di tutto ci sono i tipi:
    *classi, strutture, interfacce, enumerativi, delegati*
- Fortemente tipizzato (compile-time)
- Object-oriented
    - Campi, metodi, tipi annidati, proprietà, ...
- Overload di metodi (compile-time)
- Invocazione metodi virtuali risolta a run-time
- Ereditarietà singola di estensione
- Ereditarietà multipla di interfaccia


=== Common Language Specification

- Definisce le regole di compatibilità tra linguaggi
    (sottoinsieme di CTS)
       - Regole per gli identificatori
          - Unicode, case-sensitivity
          - Keyword
       - Regole per denominazione proprietà ed eventi
       - Regole per costruttori degli oggetti
       - Regole di overload più restrittive
       - Ammesse interfacce multiple con metodi con lo stesso nome
       - Non ammessi puntatori unmanaged
       - ...

#cfigure("images/2024-04-11-09-53-51.png",90%)

==== Tipi nativi

#cfigure("images/2024-04-11-09-54-55.png",90%)


=== Common Type System

- Tutto è un oggetto
    - `System.Object` è la classe radice
- Due categorie di tipi
    - Tipi riferimento
       - Riferimenti a oggetti allocati sull'*heap* gestito
       - Indirizzi di memoria
    - Tipi valore
       - Allocati sullo *stack* o parte di altri oggetti
       - Sequenza di byte
- Sono memorizzati come in Java



==== Tipi valore

- I tipi valore comprendono:
    - Tipi primitivi (built-in)
       - Int32, ...
       - Single, Double
       - Decimal
       - Boolean
       - Char
    - Tipi definiti dall'utente
       - Strutture (`struct`)
       - Enumerativi (`enum`)


==== Tipi valore vs tipi riferimento

#cfigure("images/2024-04-11-10-02-44.png",90%)
Appena dichiaro la variabile `r`, questa viene salvato nello Stack, poi, quando viene inizializzata, viene creato un riferimento della classe `Csize` nello Heap.
==== Common Type System


```
public struct Point
{
    private int _x, _y;
    public Point(int x, int y)
    {
        _x = x;
        _y = y;
    }
    public int X
    {
        get { return _x; }
        set { _x = value; }
    }
    public int Y
    {
        get { return _y; }
        set { _y = value; }
    }
}
```


```
public class Rectangle
{
    Point v1;
    Point v2;
    ...
}
...
Rectangle r = new Rectangle();
```

#cfigure("images/2024-04-11-10-05-02.png",90%)

```
...
Point[] points = new Point[100];
for (int i = 0; i < 100; i++)
    points[i] = new Point(i, i);
...
```
- Alla fine, rimane un solo oggetto nell'heap
    (l'array di `Point`)

```
...
Point[] points = new Point[100];
for (int i = 0; i < 100; i++)
{
    points[i].X = i;
    points[i].Y = i;
}
...
```
\
```
public class Point
{
    private int _x, _y;
    public Point(int x, int y)
    {
        _x = x;
        _y = y;
    }
    public int X
    {
        get { return _x; }
        set { _x = value; }
    }
    public int Y
    {
        get { return _y; }
        set { _y = value; }
    }
}
```

#cfigure("images/2024-04-11-10-18-32.png",90%)


```
...
Point[] points = new Point[100];
for (int i = 0; i < 100; i++)
points[i] = new Point(i, i);
...
```
- Alla fine, rimangono 101 oggetti nell'heap
    (1 array di `Point` + 100 `Point` )

```
...
Point[] points = new Point[100];
for (int i = 0; i < 100; i++)
{
    points[i].X = i;
    points[i].Y = i;
}
...
```
NO!


==== Tipi valore e tipi riferimento

#cfigure("images/2024-04-11-10-19-45.png",100%)


=== Boxing / Unboxing

- Un qualsiasi tipo valore può essere automaticamente
    convertito in un tipo riferimento (#text(blue)[*boxing*])
    mediante un up cast implicito a `System.Object`
    ```
    int i = 123;
    object o = i;
    ```

- Un tipo valore “boxed” può tornare a essere
    un tipo valore standard (#text(blue)[*unboxing*])
    mediante un down cast esplicito
    ```
    int k = (int) o;
    ```
- Un tipo valore “boxed” è un #text(blue)[*clone indipendente*], quindi se, dopo aver eseguito il boxing, cambio il valore di `i`, il valore di `k` non viene modificato

#cfigure("images/2024-04-11-10-40-03.png",90%)


=== Bibliografia


Libri di base:

- D. S. Platt, IntroducingMicrosoft® .NET, Second Edition
- J. Sharp, J. Jagger, Microsoft® Visual C\# `™` .NET Stepby Step
- T. Archer, A. Whitechapel, Inside C\#, Second Edition
- M. J. Young, XML Stepby Step, Second Edition
- R. M. Riordan, Microsoft® ADO.NET Stepby Step


Libri avanzati:

- J. Richter, AppliedMicrosoft® .NET Framework Programming
- C. Petzold, Programming Microsoft® Windows® with C\#
- S. Lidin, Inside Microsoft® .NET IL Assembler




== Garbage Collection


=== Utilizzo di un oggetto

- In un ambiente object-oriented, ogni oggetto che deve essere utilizzato dal programma
    - È descritto da un tipo
    - Ha bisogno di un'area di memoria dove memorizzare il suo stato
    - Una volta che si conosce il tipo dell'oggetto sappiamo quanta memoria allocare per lo stesso
- Passi per utilizzare un oggetto di tipo riferimento:
    - #text(blue)[*Allocare memoria*] per l'oggetto
    - #text(blue)[*Inizializzare la memoria*] per rendere utilizzabile l'oggetto
    - #text(blue)[*Usare l'oggetto*]
    - #text(blue)[*Eseguire un clean up*] dello stato dell'oggetto, se necessario; se, ad esempio, all'interno di quell'oggetto c'era un riferimento a un puntatore a file, quel file deve essere chiuso
    - #text(blue)[*Liberare la memoria*]


=== Ciclo di vita di un oggetto
#cfigure("images/2024-04-15-11-20-16.png",90%)

=== Allocazione della memoria

- In C:
    - `malloc ( calloc , realloc )`
- In C++:
    - `malloc ( calloc , realloc )`
    - `new`
- In Java:
    - `new`
- In IL:
    - `newobj`
- In C\#:
    - `new`


=== Inizializzazione della memoria

- Definite Assignment: a ogni variabile deve essere sempre assegnato un valore prima che essa venga utilizzata

- il compilatore deve assicurarsi che ciò sia sempre verificato
- Data-flow analysis del codice
- valori di default
- usati in generale per tipi valore
- ad esempio, in Java le variabili di classe, locali e i componenti di un array sono inizializzati al valore di default, non le variabili di istanza, perché in quel caso è il costruttore che si deve occupare di inizializzarle
- costruttore
- usato per i tipi classe (Java, C++, C\#)


=== Definite Assignment

```C
int k;
if (v > 0 && (k = System.in.read()) >= 0)
    System.out.println(k);
```
Questo è corretto

\
```C
int k;
while (n < 4) {
    k = n;
    if (k >= 5) break;
    n = 6;
}
System.out.println(k);
```
Questo è sbagliato, perché il compilatore non sa il valore di `n`

\
```C
int k;
while (true) {
    k = n;
    if (k >= 5) break;
    n = 6;
}
System.out.println(k);
```
Questa è corretta

\
```C
int k;
int n=5;
if (n>2) k=3;
System.out.println(k);
```
Questo non è corretto perché, a differenza dell'esempio sopra, l'espressione all'interno dell'`if` non è costante; se pensiamo a `n` come variabile condivisa, il valore di `n` potrebbe cambiare tra l'assegnamento e la valutazione.


=== Clean up dello stato

- In C++/C\#:
    - distruttore (più propriamente, finalizzatore): \~{nome della classe}
    - unico, non ereditabile, no overload, senza parametri
       e modificatori
    - invocato automaticamente alla distruzione dell'oggetto
       (non può essere invocato)
- In java:
    - `finalize()`
    - metodo di `Object`
    - invocato automaticamente alla distruzione dell'oggetto
       (non può essere invocato)
    - il momento in cui viene invocato un finalizzatore dipende
       dalla JVM


=== Liberazione della memoria

- In C:
    - `free()`
- In C++:
    - `free()`
    - `delete`
- In java/C\#: garbage collector (GC)


=== Cos'è il Garbage Collection

- Modalità automatica di rilascio delle risorse utilizzate da un oggetto

- Migliora la stabilità dei programmi
    - Evita errori connessi alla necessità, da parte del programmatore,
       di manipolare direttamente i puntatori alle aree di memoria
- Pro:
    - dangling pointer: puntatore che fa riferimento a un'area di memoria non più valida
    - doppia de-allocazione
    - memory leak
- Contro:
    - aumentata richiesta risorse di calcolo
    - incertezza del momento in cui viene effettuata la GC
    - rilascio della memoria non deterministico; quindi non posso distruggere un oggetto quando voglio io


- Strategie disponibili:
    - Tracing
        - determinare quali oggetti sono (potenzialmente) raggiungibili
        - eliminare gli oggetti non raggiungibili
    - Reference counting
        - ogni oggetto contiene un contatore che indica
        il numero di riferimenti a esso
        - la memoria può essere liberata quando il contatore raggiunge lo 0
    - Escape analysis
        - si spostano oggetti dallo heap allo stack
        - l'analisi viene effettuata a compile-time in modo da stabilire se un oggetto, allocato all'interno di una subroutine, non è accessibile al di fuori di essa
        - riduce il lavoro del GC


=== GC: Reference counting

- Svantaggi:
    - Cicli di riferimenti
       - se due oggetti si referenziano a vicenda, il loro contatore
          non raggiungerà mai 0
    - Aumento dell'occupazione di memoria
    - Riduzione della velocità delle operazioni sui riferimenti
       - ogni operazione su un riferimento deve anche incrementare/decrementare
          i contatori
    - Atomicità dell'operazione
       - ogni modifica a un contatore deve essere resa operazione atomica
          in ambienti multi-threaded
    - Assenza di comportamento real-time
       - ogni operazione sui riferimenti può (potenzialmente)
          causare la de-allocazione di diversi oggetti
       - il numero di tali oggetti è limitato solamente dalla memoria allocata


=== GC: Tracing

- Siano #text(aqua)[p] e #text(aqua)[q] due oggetti
- Sia #text(aqua)[q] un oggetto raggiungibile
- Diremo che p è raggiungibile in maniera ricorsiva se e solo se:
    - esiste un riferimento a #text(aqua)[p] tramite #text(aqua)[q]
    - ovvero #text(aqua)[p] è raggiungibile attraverso un oggetto, a sua volta raggiungibile
- Un oggetto è pertanto raggiungibile in due soli casi:
- è un oggetto radice
    - creato all'avvio del programma (oggetto globale)
    - creato da una sub-routine (oggetto scope, riferito da variabile sullo stack)
- è referenziato da un oggetto raggiungibile
    - la raggiungibilità è una chiusura transitiva


- La definizione di garbage tramite la raggiungibilità non è ottimale
    - può accadere che un programma utilizzi per l'ultima volta un certo oggetto molto prima che questo diventi irraggiungibile
- Distinzione:
    - garbage *sintattico* \ (oggetti che il programma non può raggiungere)
    - garbage *semantico* \ (oggetti che il programma non vuolepiù usare)
        - problema solo parzialmente decidibile, quindi non possono essere creati algoritmi per risolvere questo problema
```
Object x = new Foo();
Object y = new Bar();
x = new Quux(); // qui l'oggetto Foo è garbage sintattico
if(x.check_something())
x.do_something(y); // qui y *potrebbe* essere garbage
semantico
```

=== Allocazione della memoria

- In fase di inizializzazione di un processo, il CLR
    - Riserva una regione contigua di spazio di indirizzamento
       managedheap
    - Memorizza in un puntatore (`NextObjPtr`) l'indirizzo di partenza
       della regione

#cfigure("images/2024-04-15-11-42-57.png",90%)


- Quando deve eseguire una `newobj`, il CLR
    - Calcola la dimensione in byte dell'oggetto e aggiunge all'oggetto
       due campi di 32 (o 64) bit
          - Un puntatore alla tabella dei metodi
          - Un campo `SyncBlockIndex`
    - Controlla che ci sia spazio sufficiente a partire da `NextObjPtr`
       - in caso di spazio insufficiente:
          - garbage collection
          - `OutOfMemoryException`
    - `thisObjPtr= NextObjPtr;`
    - `NextObjPtr += sizeof(oggetto);`
    - Invoca il costruttore dell'oggetto ( `this` $equiv$ `thisObjPtr` )
    - Restituisce il riferimento all'oggetto



#cfigure("images/2024-04-15-11-47-22.png",90%)

=== Garbage Collector

- Verifica se nell'heap esistono oggetti non più utilizzati dall'applicazione

- Ogni applicazione ha un insieme di radici (_root_)
- Ogni radice è un puntatore che contiene l'indirizzo di un oggetto
    di tipo riferimento oppure vale `null`
- Le radici sono:
    - Variabili globali e field statici di tipo riferimento
    - Variabili locali o argomenti attuali di tipo riferimento sugli stack
       dei vari thread
    - Registri della CPU che contengono l'indirizzo di un oggetto
       di tipo riferimento
- *Gli oggetti “vivi”* sono quelli *raggiungibili* direttamente
    o indirettamente dalle radici
- *Gli oggetti garbage* sono quelli *NON raggiungibili*
    direttamente o indirettamente dalle radici


- Quando parte, il GC ipotizza che tutti gli oggetti siano garbage
- Quindi, scandisce le radici e per ogni radice *marca*
    - l'eventuale oggetto referenziato e
    - Tutti gli oggetti a loro volta raggiungibili a partire da tale oggetto
- Se durante la scansione incontra un oggetto già marcato in precedenza, lo salta
    - sia per motivi di prestazioni
    - sia per gestire correttamente riferimenti ciclici
- Una volta terminata la scansione delle radici, tutti gli oggetti NON marcati sono non raggiungibili e quindi garbage

==== Fase 1: Mark

#cfigure("images/2024-04-15-11-50-46.png",90%)


- Rilascia la memoria usata dagli oggetti non raggiungibili
- *Compatta* la memoria ancora in uso, *modificando nello stesso tempo tutti i riferimenti agli oggetti spostati!* Quindi sposta tutti gli oggetti in ordine dall'inizio dell'heap.

- Unifica la memoria disponibile, aggiornando il valore di `NextObjPtr`
- Tutte le operazioni che il GC effettua sono possibili in quanto
    - Il tipo di un oggetto è sempre noto
    - È possibile utilizzare i metadati per determinare quali field
        dell'oggetto fanno riferimento ad altri oggetti


==== Fase 2: Compact

#cfigure("images/2024-04-15-11-52-49.png",90%)

=== Finalization

- Non è responsabilità del GC, ma del programmatore
- Se un oggetto contiene esclusivamente
    - tipi valore e/o
    - riferimenti a oggetti managed
 (maggior parte dei casi), non è necessario eseguire alcun codice particolare

- Se un oggetto contiene almeno un riferimento a un oggetto _unmanaged_ (in genere, una risorsa del S.O.)
- file, connessione a database, socket, mutex, bitmap, ...\
 #text(blue)[*è necessario eseguire del codice per rilasciare la risorsa, prima della deallocazione dell'oggetto*]


- Ad esempio, un oggetto di tipo `System.IO.FileStream`
    - Prima deve aprire un file e memorizzare in un suo field l'handle del file (una risorsa di S.O. unmanaged)
    - Quindi usa tale handle nei metodi *Read* e *Write*
    - Infine, deve rilasciare l'handle nel metodo `Finalize`
- In C\#
    - NON è possibile definire il metodo `Finalize`
    - È necessario definire un *distruttore* (sintassi C++)

```
public class OSHandle
{
//' Field contenente l'handledella risorsa unmanaged
private readonly IntPtr handle;


public IntPtr Handle
{ get { return _handle; } }

public OSHandle(IntPtr handle)
{ _handle = handle; }

~OSHandle()
{ CloseHandle(_handle); }

[System.Runtime.InteropServices.DllImport(“Kernel32”)]
private extern static bool CloseHandle(IntPtr handle);
}
```


- Il compilatore C\# trasforma il codice del distruttore

```
~OSHandle()
{ CloseHandle(_handle); }
```
 nel seguente codice (ovviamente in IL):

```
protected override void Finalize()
{
    try
    { CloseHandle(_handle); }
    finally
    { base.Finalize(); }
}
```

- L'invocazione del metodo `Finalize` non avviene in modo deterministico
- Inoltre, non essendo un metodo pubblico, il metodo `Finalize` non può essere invocato direttamente
- Nel caso di utilizzo di risorse che devono essere rilasciate appena termina il loro uso, questa situazione è problematica
- Si pensi a file aperti o a connessioni a database che vengono chiusi solo quando il GC invoca il corrispondente metodo `Finalize`

- In questi casi, è di fondamentale importanza rilasciare (`Dispose`) o chiudere (`Close`) la risorsa in modo deterministico


=== Rilascio deterministico senza gestione delle eccezioni

```
...
Byte[] bytesToWrite = new Byte[] {1,2,3,4,5};
FileStream fs;
fs = new FileStream(“Temp.dat”, FileMode.Create);
fs.Write(bytesToWrite, 0, bytesToWrite.Length);
fs.Close();
...
```

=== Rilascio deterministico con gestione delle eccezioni

```
...
Byte[] bytesToWrite = new Byte[] {1,2,3,4,5};
FileStream fs = null;
try
{
    fs = new FileStream(“Temp.dat”, FileMode.Create);
    fs.Write(bytesToWrite, 0, bytesToWrite.Length);
}
finally
{
    if(fs != null) fs.Close();
}
...
```

=== Il pattern Dispose

 Se un tipo `T` vuole offrire ai suoi utilizzatori un servizio di clean up esplicito, deve implementare l'interfaccia `IDisposable`

```
public interface IDisposable
{
    void Dispose();
}
```
- I clienti di T possono utilizzare l'istruzione `using`

```
using (T tx = ...)
{
    utilizzo di tx... #Invocazione automatica di tx.Dispose()
}
```

=== Rilascio deterministico con using

```
...
Byte[] bytesToWrite = new Byte[] {1,2,3,4,5};
using (FileStream fs = new FileStream(“Temp.dat”, FileMode.Create))
{
    fs.Write(bytesToWrite, 0, bytesToWrite.Length);
}
...
```
- Il tipo della variabile definita nella parte iniziale di `using` deve implementare l'interfaccia IDisposable

- All'uscita del blocco 'using' , viene sempre invocato automaticamente il metodo Dispose


=== Il pattern Dispose (altro esempio di utilizzo)

```
public class CursorReplacer : IDisposable
{
    private readonly Cursor _previous;
    public CursorReplacer()
    {
        _previous = Cursor.Current;
        Cursor.Current = Cursors.WaitCursor;
    }
    public void Dispose()
    {
        Cursor.Current = _previous;
    }
}
```

```
List<DbTableWrapper> tableWrappers = new List<DbTableWrapper>();
// Recupero di tutte le tabelle selezionate
using (CursorReplacer cursorReplacer = new CursorReplacer())
{
    foreach (DbServerWrapper serverWrapper in SelectedDbServerWrappers)
        foreach (DbCatalogWrapper catalogWrapper in serverWrapper.SelectedDbCatalogWrappers)
            foreach (DbTableWrapper tableWrapper in catalogWrapper.SelectedDbTableWrappers)
        {
            tableWrappers.Add(tableWrapper);
        }
}
```



== Tipi in .NET

- Dal punto di vista del modo in cui le istanze vengono gestite
    in memoria (rappresentazione, tempo di vita, ...),
    i tipi possono essere distinti in:
       - _Reference type_
       - _Value type_
- Dal punto di vista sintattico (sintassi del linguaggio C\#),
    i tipi possono essere distinti in:
       - Classi - `class`
       - Interfacce - `interface`
       - Strutture - `struct`
       - Enumerativi - `enum`
       - Delegati - `delegate`
       - Array - `[]`
- In .NET, si concretizzano sempre in una classe (anche nel caso di tipi built-ine di interfacce)


- In generale, un tipo può contenerela definizione di 0+:
    - #highlight(fill: myblue)[Costanti] - sempre implicitamente associate al tipo
    - #highlight(fill: myblue)[Campi] (field) - read-only o read-write, associati alle istanze o al tipo
    - #highlight(fill: myblue)[Metodi] - associati alle istanze o al tipo
    - #highlight(fill: myblue)[Costruttori] - di istanza o di tipo
    - #highlight(fill: myblue)[Operatori] - sempre associati al tipo
    - #highlight(fill: myblue)[Operatori di conversione] - sempre associati al tipo
    - #highlight(fill: myblue)[Proprietà] - associate alle istanze o al tipo
    - #highlight(fill: myblue)[Indexer] - sempre associati alle istanze
    - #highlight(fill: myblue)[Eventi] - associati alle istanze o al tipo
    - #highlight(fill: myblue)[Tipi] - annidati


=== Modificatori di visibilità

#cfigure("images/2024-04-17-10-02-56.png",100%)

#cfigure("images/2024-04-17-17-46-42.png",100%)


- Non sono applicabili nei seguenti casi:
    - #highlight(fill: myblue)[Costruttori di tipo] (statici)
       sempre inaccessibili - invocati direttamente dal CLR
    - #highlight(fill: myblue)[Distruttori] (finalizer)
       sempre inaccessibili - invocati direttamente dal CLR
    - #highlight(fill: myblue)[Membri di interfacce]
       sempre pubblici
    - #highlight(fill: myblue)[Membri di enum]
       sempre pubblici
    - #highlight(fill: myblue)[Implementazione esplicita di membri di interfacce]
       visibilità particolare (pubblici/privati), non modificabile
    - #highlight(fill: myblue)[Namespace]
       sempre pubblici


=== Regole

- *Massimizzare l'incapsulamento minimizzando la visibilità*

- Information hiding a livello di assembly
    - Dichiarare `public` solo i tipi significativi dal punto di vista concettuale
- Information hiding a livello di classe
    - Dichiarare `public` solo metodi, proprietà ed eventi significativi
       dal punto di vista concettuale
    - Dichiarare `protected` solo le funzionalità che devono essere visibili
       nelle classi derivate, ma non esternamente
       ad esempio, costruttori particolari, metodi e proprietà virtuali
       non public
- Information hidinga livello di field
    - Field `private` e proprietà `public`
    - Field `private` e proprietà `protected`


=== Costanti

- Una *costante* è un simbolo che identifica un valore
    che non può cambiare
- Il *tipo* della costante può essere solo un tipo considerato primitivo
    dal CLR (compreso `string`)
- Il *valore* deve essere determinabile _a tempo di compliazione_
- Ad esempio, in `Int32` esistono:

```
public const int MaxValue = 2147483647;
public const int MinValue = -2147483648;
```
- In una classe contenitore di dimensioni prefissate, si potrebbe definire:
    ```
    public const int MaxEntries = 100; // Warning!
    ```

- Si noti l'utilizzo della maiuscola iniziale
- È possibile applicare `const` anche alle variabili locali


=== Field

- Un *field* è un _data member_ che può contenere:
    - un *valore* (un istanza di un _value type_)
    - un *riferimento* (a un'istanza di un _reference type_) in genere, la realizzazione di un'associazione
- Può essere:
    - di *istanza* (_default_)
    - di *tipo* (`static`)
- Può essere:
    - *read-write* (_default_)
    - *read-only* (`readonly`) \
       inizializzato nella definizione o nel costruttore
- Esiste sempre un *valore di default* (`0 , 0.0, false, null`)



- Qual è la differenza tra le seguenti definizioni:

```
public const int MaxEntries = 100;
public static readonly int MaxEntries = 100;
```

- Nel primo caso, la costante `MaxEntries` viene *“iniettata”* nel codice
    del cliente
       - se il valore viene modificato e se il cliente e il fornitore sono in assembly diversi, *è necessario ricompilare anche il codice del cliente*
- Nel secondo caso, l'accesso al `fieldMaxEntries` è quello standard:
    il valore è in memoria ed è necessario reperirlo
       - se il valore viene modificato e se il cliente e il fornitore sono in _assembly_ diversi,
          *NON è necessario ricompilare anche il codice del cliente*


=== Regole

- Definire `const` solo le costanti *“vere”*, cioè i valori
    veramente immutabili nel tempo (nelle versioni del programma),
    negli altri casi utilizzare field statici read-only
       - il valore di `MaxEntries` non è una costante “vera”
          perché in una versione successiva del programma potrebbe cambiare
- *Costanti*
    - il nome dovrebbe iniziare con una lettera maiuscola
    - di solito, dovrebbe essere pubblica (ma non è sempre così)
- *Field*
    - il nome dovrebbe iniziare con “\_” seguito da una lettera minuscola
    - deve essere privata (accesso sempre mediante proprietà)
- *Field read-only*
    - scegliere, a seconda delle situazioni,
       una delle due convenzioni precedenti


=== Modificatori di metodi

- `virtual`
- `abstract`
- `override`
- `override sealed / sealed override`
- Applicabili a:
    - *Metodi*
    - *Proprietà* (metodi `get` e `set`)
    - *Indexer* (metodi `get` e `set`)
    - *Eventi* (metodi `add` e `remove`)
 di istanza (cioè non statici)


=== Modificatore `virtual`

- *L'implementazione di un metodo virtuale può essere modificata* da un membro override di una classe derivata (discendente)

- Quando il metodo virtuale viene invocato, viene valutato il tipo run-time dell'oggetto su cui è invocato per vedere la presenza di un membro sovrascritto
    - Late binding
    - Polimorfismo
- *Per default, i metodi non sono virtuali*

```
protected virtual void Method()
{ ... }
public virtual int Property
{ get { ... } set { ... } }
public virtual int this[int index]
{ get { ... } }
```

=== Modificatore `abstract`

- Si usa il modificatore `abstract` per indicare che il metodo non contiene alcuna implementazione

- I metodi astratti hanno le seguenti caratteristiche
    - Un metodo virtuale è implicitamente virtuale
    - La dichiarazione di metodi astratti è permessa solo in classi astratte
- L'implementazione di un metodo astratto verrà fornita da un metodo sovrascrivente

```
protected abstract void Method();
public abstract int Property
{ get; set; }
public abstract int this[int index]
{ get; }
```

=== Modificatore `override`

- Un metodo override *fornisce una (nuova) implementazione*
    di un metodo ereditato da una classe base
       - Il metodo sovrascritto da una dichiarazione override
          è detto metodo basesovrascitto(overridden)
- Il metodo base sovrascritto
    - Deve essere *virtual*, *abstract*, o *override*
    - Deve avere la stessa firma (signature) del metodo override
- Una dichiarazione override *non può cambiare l'accessibilità*
    del metodo base sovrascritto (diverso da Java)
- L'uso del modificatore `sealed` impedisce a una qualsiasi classe derivata
    l'ulteriore sovrascrittura del metodo
    ```
    protected override void Method()
    { ... }
    public override sealed int Property
    { get { ... } set { ... } }
    public override int this[int index]
    { get { ... } }
    ```

=== Metodi

====  Passaggio degli argomenti

- Tre tipi di argomenti:
    - *In* (`default` in C\#)
       - L'argomento deve essere inizializzato
       - L'argomento viene passato per *valore* (per *copia*)
       - Eventuali modifiche del valore dell'argomento
          *non hanno effetto* sul chiamante
    - *In/Out* (`ref` in C\#)
       - L'argomento deve essere inizializzato
       - L'argomento viene passato per *riferimento*
       - Eventuali modifiche del valore dell'argomento *hanno effetto* sul chiamante
    - *Out* (`out` in C\#)
       - L'argomento può NON essere inizializzato
       - L'argomento viene passato per *riferimento*
       - Le modifiche del valore dell'argomento (l'inizializzazione è obbligatoria)
          *hanno effetto* sul chiamante


====  Passaggio degli argomenti In

- *Value type*
    - Viene passata una copia dell'oggetto
    - Eventuali modifiche vengono effettuate sulla copia
       e *non hanno alcun effetto* sull'oggetto originale
- *Reference type*
    - Viene passata una copia del riferimento all'oggetto
    - Eventuali modifiche dell'oggetto referenziato hanno effetto
    - Eventuali modifiche del riferimento vengono effettuate sulla copia
       e *non hanno alcun effetto* sul riferimento originale

```
Point p1 = new Point(0,0);
Method1(p1);
Console.WriteLine("{0}",p1);

static void Method1(Point p)
{
    p.X = 100; p.Y = 100;
}
```
- Se `Point` è una classe le modifiche vengono effettuate sull'oggetto referenziato\
    `(100,100)`
- Se `Point` è una struttura le modifiche vengono effettuate sulla copia\
    `(0,0)`


====  Passaggio degli argomenti In/Out

- *Value type*
    - Viene passato l'indirizzo dell'oggetto
    - Eventuali modifiche *agiscono direttamente sull'oggetto* originale
- *Reference type*
    - Viene passato l'indirizzo del riferimento all'oggetto
    - Eventuali modifiche dell'oggetto referenziato *hanno effetto*
    - Eventuali modifiche del riferimento
       *agiscono direttamente sul riferimento originale*
```
Point p1 = new Point(0,0);
Method2(ref p1);
Console.WriteLine("{0}",p1);

static void Method2(ref Point p)
{
    p.X = 100; p.Y = 100;
}
```
- Se `Point` è una classe\
    `(100,100)`
- Se `Point` è una struttura\
    `(100,100)`


====  Passaggio degli argomenti
```C
class/struct Persona ...

...
Persona p1 = new Persona(“Tizio”); // p1 == Tizio*

Method1(p1);

// p1 == Tizio

Method2(ref p1);

// p1 == Sempronio
...
static void Method1(Persona p)
{
    p = new Persona(“Caio”); //* p == Caio
}

static void Method2(ref Persona p)
{
    p = new Persona(“Sempronio”); //* p == Sempronio
}
```
- `p1` all'inizio fa riferimento a un'istanza di `Persona` con nome "Tizio"
- quando viene invocato ```c Method1()``` nello scope della funzione la variabile `p` fa riferimento a un'istanza di `Persona` di nome "Caio", ma quando si esce dalla funzione il record di attivazione viene distrutto e quindi nel `main` `p1` fa ancora riferimento all'istanza di `Persona` di prima, di nome "Tizio"
- Nella funzione ```c Method2()``` viene creato un riferimento tra `p` e `p1`, cioè la variabile `p` punta alla variabile `p1`, che a sua volta punta all'istanaza di `Persona` "Tizio"; quindi quando si invoca il metodo ```c new Persona()```, siccome tutte le operazioni fatte su `p` vengono fatte su `p1`, è la variabile `p1` che punta a una nuova istanza di `Persona` "Sempronio"; quindi anche quando si esce dalla funzione e il record di attivazione viene distrutto, la variabile `p1` continua a puntare all'istanza di `Persona` "Sempronio"

====  Passaggio degli argomenti Out

- *Value type* e *Reference type*
    - Viene passato l'indirizzo dell'oggetto o del riferimento all'oggetto
       come nel caso In/Out
    - Non è necessario che l'oggetto o il riferimento siano inizializzati
       prima di essere passati come argomento
    - L'oggetto o il riferimento *DEVONO essere inizializzati*
       nel metodo a cui sono stati passati come argomento
```
Point p1;
Method3(out p1);

static void Method3(out Point p)
{
    // In questo punto il compilatore suppone che
    // p NON sia inizializzato
    p.X = 100; p.Y = 100; // Errore di compilazione!
    p = new Point(100,100); // È indispensabile
}
```

==== Regole

- Utilizzare prevalentemente il passaggio _standard_ per valore
- Utilizzare il passaggio per riferimento (`ref` o `out`) solo se strettamente necessario
    - 2+ valori da restituire al chiamante
    - 1+ valori da utilizzare e modificare nel metodo
    - Scegliere `ref` se l'oggetto passato come argomento
        deve essere già stato inizializzato
    - Scegliere `out` se è responsabilità del metodo
        inizializzare completamente l'oggetto passato come argomento


==== Esempi
```C
public static void Swap<T>(ref T arg1, ref T arg2)
{
    T temp = arg1;
    arg1 = arg2;
    arg2 = temp;
}

public static void SplitCognomeNome(string cognomeNome,
    out string cognome, out string nome)
{
    string[] words = cognomeNome.Split(' ');
    if (words.Length == 2)
    {
        cognome = words[0];
        nome = words[1];
    }
    else
    {
        ...
    }
}
```


====  Numero variabile di argomenti

- Si supponga di dover scrivere:

```C
Add(a,b); // a+b
Add(10,20,30); // 10+20+30
Add(x1,x2,x3,x4); // x1+x2+x3+x4
```
- Soluzioni possibili:
    - Overloadingdel metodo *`Add`*
       - *Svantaggio*: posso solo codificare un numero finito di metodi
    - Definire un solo metodo Addche accetti un numero variabile
       di argomenti

```C
int Add(params int[] operands)
{
    int total = 0;
    foreach (int operand in operands)
        total += operand;
    return total;
}
```


- Non solo posso scrivere:

```C
Add(a,b);
Add(10,20,30);
Add(x1,x2,x3,x4);
```
- Ma anche:

```C
Add(); // restituisce 0
int[] numbers = { 10,20,30,40,50 };
Add(numbers);
Add(new int[] { 10,20,30,40,50 });
Add(new int[] { x1,x2,x3,x4,x5 });
```
- Zucchero sintattico:

```C
Add(x1,x2,x3,x4);
```
è uguale a
```C
Add(new int[] { x1,x2,x3,x4 });
```

=== Costruttori di istanza

- *Responsabilità*: inizializzare correttamente lo stato dell'oggetto appena creato (nulla di più!)

- In mancanza di altri costruttori, esiste sempre un costruttore di _default_ senza argomenti che, semplicemente, invoca il costruttore senza argomenti della classe base

- Nel caso delle *classi*, il costruttore senza argomenti può essere definito dall'utente

- Nel caso delle *strutture*, il costruttore senza argomenti NON può essere definito dall'utente (per motivi di efficienza)

- In entrambi i casi, è possibile definire altri costruttori

con differente signaturee differente visibilità

```C
public abstract class DataAdapterManager
{
    private readonly IDataTable _dataTable;
    private readonly IConnectionManager _connectionManager;

    protected DataAdapterManager(IDataTable dataTable,
    IConnectionManager connectionManager)
    {
        if(dataTable == null)
        throw new ArgumentNullException("dataTable");
        if(connectionManager == null)
        throw new ArgumentNullException("connectionManager");
        _dataTable = dataTable;
        _connectionManager = connectionManager;
    }
    ...
}
```


```C
public class XmlDataAdapterManager : DataAdapterManager
{
private readonly Encoding _encoding;


public XmlDataAdapterManager(IDataTable dataTable,
    XmlConnectionManager xmlConnectionManager)
: this(dataTable, xmlConnectionManager, Encoding.Unicode)
{ }

public XmlDataAdapterManager(IDataTable dataTable,
    XmlConnectionManager xmlConnectionManager,
    Encoding encoding)
: base(dataTable, xmlConnectionManager)
{
_encoding = encoding;
}
...}
```

=== Costruttori di tipo

- *Responsabilità*: inizializzare correttamente lo stato comune a tutte le istanze della classe - _field_ statici

- Dichiarato `static`
- Implicitamente `private`
- Sempre senza argomenti - no _overloading_
- Può accedere esclusivamente ai membri (_field_, metodi, ...) statici della classe

- Se esiste, viene invocato automaticamente dal CLR
    - Prima della creazione della prima istanza della classe
    - Prima dell'invocazione di un qualsiasi metodo statico della classe
- Non basare il proprio codice sull'ordine di invocazione di costruttori di tipo


```C
class MyType
{
    static int _x = 5; #Viene definito implicitamente un costruttore di tipo
    ...
}

class MyType
{
    static int _x;
    static MyType() { _x = 5; } #Del tutto analogo al caso precedente
    ...
}

class MyType
{
    static int _x = 5;
    static MyType() { _x = 10; } #_x viene prima inizializzato a 5 e quindi a 10
    ...
}
```

=== Regole

- Definire un costruttore di tipo solo se strettamente necessario, cioè se i campi statici della classe
    - NON possono essere inizializzati in linea
    - Devono essere inizializzati solo se la classe
        viene effettivamente utilizzata

```C
public class A
{
    private static XmlDocument _xmlDocument;

    static A()
    {
        _xmlDocument = new XmlDocument();
        _xmlDocument.Load (...);
    }
    ...
}
```

=== Costruttori ed eccezioni

- Supponiamo che
    - un costruttore lanci un'eccezione e
    - l'eccezione non venga gestita all'interno del costruttore stesso
       (quindi arrivi al chiamante)
- *Nel caso di costruttori di istanza* nessun problema!\
    In C++ è una situazione non facilmente gestibile

- *Nel caso di costruttori di tipo* la classe NON è più utilizzabile!\
    `TypeInitializationException`


=== Interfacce

- In C\#, un'interfaccia può contenere esclusivamente:
    - *Metodi* - considerati pubblici e astratti
    - *Proprietà* - considerate pubbliche e astratte
    - *_Indexer_* - considerati pubblici e astratti
    - *Eventi* - considerati pubblici e astratti
- In CLR, un'interfaccia è considerata una particolare classe astratta
    di sistema che (ovviamente) non deriva da `System.Object`
       - però, le classi che la implementano derivano per forza
          da `System.Object`
- Un'interfaccia
    - Può essere implementata sia dai _reference type_, sia dai _value type_
    - È considerata sempre un *reference type*
    - *Attenzione*: se si effettua il cast di un _value type_ a un'interfaccia,
       avviene un *boxing del value type* (con conseguente copia del valore)!


====  Implementazione di un'interfaccia

```C
public interface IBehavior
{
    void Method();
    int Property { get; set; }
    int this[int index] { get; }
}

public class A : IBehavior
{
    public void Method() // virtual sealed
    { ... }
    public int Property // virtual sealed
    {
        get { ... }
        set { ... }
    }
    public int this[int index] // virtual sealed
    {
        get { ... }
    }
}
```

```C
public class A : IBehavior
{
    public virtual void Method()
    { ... }
    public virtual int Property
    {
        get { ... }
        set { ... }
    }
    public virtual int this[int index]
    {
        get { ... }
    }
}

public class B : A
{
    public override void Method() ...
    public override int Property ...
    public override int this[int index ] ...
}```


====  Implementazione di un'interfaccia: classe astratta

```C
public abstract class A : IBehavior
{
    public abstract void Method();
    public abstract int Property { get; set; }
    public abstract int this[int index] { get; }
}

public class B : A
{
    public override void Method() ...
    public override int Property ...
    public override int this[int index] ...
}```
- Siccome `A` è una classe astratta, la sottoclasse `B` che la estende deve ridefinire tutti i metodi astratti di `A`

====  Implementazione esplicita di un'interfaccia

```C
public class A : IBehavior
{
    void IBehavior.Method()
    { ... }
    int IBehavior.Property
    {
        get { ... }
        set { ... }
    }
    int IBehavior.this[int index]
    {
        get { ... }
    }
}

A a = new A(...);
a.Method(); // Non compila!
((IBehavior) a).Method(); // Ok!
```
- Questo per
    - Name hiding
    - Avoiding name ambiguity

\
```C
public interface IMyInterface1
{ void Close(); }

public interface IMyInterface2
{ void Close(); }

public class MyClass : IMyInterface1, IMyInterface2
{
    void IMyInterface1.Close()
    {...}
    void IMyInterface2.Close()
    {...}
    public void Close()
    {...}
}

MyClass a = new MyClass (...);
((IMyInterface1) a).Close(); // Ok!
((IMyInterface2) a).Close(); // Ok!
a.Close(); // Ok!
```

#pagebreak()
=== Interfaccia vs. Classe astratta

#cfigure("images/2024-04-17-10-57-57.png",100%)

#cfigure("images/2024-04-17-10-58-30.png",100%)

#cfigure("images/2024-04-17-10-59-03.png",100%)

#cfigure("images/2024-04-17-10-59-31.png",100%)


== Classi e Interfacce Base

=== Framework .NET: Overview

#cfigure("images/2024-04-18-13-06-04.png",100%)


=== `System.Object`

- La radice della gerarchia dei tipi
    - tutte le classi nel framework.NET Framework
       sono derivate da `Object`
- Ogni metodo definito nella classe `Object` è disponibile in tutti gli oggetti del sistema

#cfigure("images/2024-04-18-13-06-35.png",90%)


- Le classi derivate possono (e/o devono) sovrascrivere (override) alcuni di tali metodi, tra cui:
    - `Equals` - Supporta il confronto tra oggetti
    - `ToString` - Crea una stringa “comprensibile”che descrive un'istanza della classe
    - `GetHashCode` - Genera un numero corrispondente al valore (stato) dell'oggetto per consentire l'uso di tabelle hash
    - `Finalize` - Effettua operazioni di “pulizia”prima che un oggetto venga automaticamente distrutto


=== `Object.Equals`

- `public virtual bool Equals(object obj);`
- Valore di ritorno: `true` se `this` è uguale a `obj`, cioè se referenziano lo stesso oggetto, altrimenti false

#cfigure("images/2024-04-18-13-10-43.png",90%)

- Le seguenti condizioni devono essere `true` per tutte le implementazioni del metodo Equals\
    Nell'elenco, `x`, `y`, e `z` rappresentano riferimenti non nulli a oggetti
    - `x.Equals(x)` restituisce true
    - `x.Equals(y)` restituisce lo stesso valore di `y.Equals(x)`
    - `x.Equals(y)` restituisce truese sia x che y sono `NaN`
    - `(x.Equals(y) && y.Equals(z))` restituisce `true`
        se e solo se `x.Equals(z)` restituisce `true`
    - Chiamate successive a `x.Equals(y)` restituiscono sempre
        lo stesso valore fintantoché gli oggetti rteferenziatida `x` e `y`
        non vengono modificati
    - `x.Equals` (un riferimento `null`) restituisce `false`
- Le implementazioni di `Equals` non devono generare eccezioni


- I tipi che sovrascrivono `Equals` devono anche sovrascrivere `GetHashCode`; altrimenti, `Hashtable` potrebbe non funzionare correttamente

- Se il linguaggio permette l'overloading di operatori e, per un certo tipo, si effettua l'overloading dell'operatore di uguaglianza, tale tipo deve anche sovrascrivere il metodo `Equals` Tale implementazione del metodo `Equals` *deve restituire gli stessi risultati dell'operatore di uguaglianza*


```C
public class Point
{
    private readonly int _x, _y;
    ...
    public override bool Equals(object obj)
    {
    //Check for null and compare run-time types.
    if(obj == null || GetType() != obj.GetType())
        return false;
    Point p = (Point) obj;
    return (_x == p._x) && (_y == p._y);
    }
    public override int GetHashCode()
    {
        return _x ^ _y;
    }
}```


=== Object.Equals

```C
public class SpecialPoint : Point
{
    private readonly int _w;
    ...
    public SpecialPoint(int x, int y, int w) : base(x, y)
    {
        _w = w;
    }
    public override bool Equals(object obj)
    {
        return base.Equals(obj) &&
        _w == ((SpecialPoint) obj)._w;
    }
    public override int GetHashCode()
    {
        return base.GetHashCode() ^ _w;
    }
}```


```C
public class Rectangle
{
    private readonly Point _a, _b;
    ...
    public override bool Equals(object obj)
    {
        if(obj == null || GetType() != obj.GetType())
        return false;
        Rectangle r = (Rectangle) obj;
        // Uses Equals to compare variables.
        return _a.Equals(r._a) && _b.Equals(r._b);
    }
    public override int GetHashCode()
    {
        return _a.GetHashCode() ^ _b.GetHashCode();
    }
}```



```C
public struct Complex
{
    private readonly double _re, _im;
    ...
    public override bool Equals(object obj)
    {
        return obj is Complex && this == (Complex) obj;
    }
    public override int GetHashCode()
    {
        return _re.GetHashCode() ^ _im.GetHashCode();
    }
    public static bool operator ==(Complex x, Complex y)
    {
        return x._re == y._re && x._im == y._im;
    }
    public static bool operator !=(Complex x, Complex y)
    {
        return !(x == y);
    }
}```


=== `System.ValueType`

#cfigure("images/value-type.jpg",100%)

#pagebreak()
=== `System.Boolean`

#cfigure("images/2024-04-18-13-21-43.png",100%)

=== `System.Int32`

#cfigure("images/2024-04-18-13-22-21.png",100%)

=== `System.IComparable`

- Confronta l'istanza corrente con un altro oggetto dello stesso tipo
- *Valore di ritorno*: un 32-bit signed integer che indica
    l'ordine relativo degli oggetti confrontati
- La semantica del valore restituito è la seguente:
    - Minore di zero - l'istanza `this` precede `obj`
    - Zero - l'istanza `this` è uguale a `obj`
    - Maggiore di zero - l'istanza `this` segue `obj`
- Per definizione, ogni oggetto segue un riferimento `null`
- Il parametro `obj` deve essere dello stesso tipo della classe
    o valuetype che implementa questa interfaccia;
    altrimenti, va lanciata una `ArgumentException`

#cfigure("images/2024-04-18-13-23-39.png",60%)

=== `System.IComparable`

- Note per gli Implementatori:

Per ogni oggetto `A`, `B` e `C`, devono valere le seguenti condizioni:

- `A.CompareTo(A)` deve restituire zero
- Se `A.CompareTo(B)` restituisce zero
    allora anche `B.CompareTo(A)` deve restituire zero
- Se `A.CompareTo(B)` restituisce zero e `B.CompareTo(C)`
    restituisce zero allora anche `A.CompareTo(C)` deve restituire zero
- Se `A.CompareTo(B)` restituisce un valore diverso da zero
    allora `B.CompareTo(A)` deve restituire un valore dal segno opposto
- Se `A.CompareTo(B)` restituisce un valore x diverso da zero,
    e `B.CompareTo(C)` un valore `y` dello stesso segno di `x`,
    allora `A.CompareTo(C)` deve restituire un valore dello stesso segno
    di `x` e `y`


- Se volessi:
    - Ordinare i punti in ordine decrescente
    - Ordinare dei film
       - Per genere, oppure
       - Per titolo
    - Ordinare degli studenti
       - Per cognome e nome, oppure
       - Per matricola, oppure
       - Per corso di studio
    - ...


====  `System.Collections.IComparer`

- Questa interfaccia è usata, ad esempio,
    dai metodi `Array.Sort` e `Array.BinarySearch`
- Fornisce un modo per personalizzare il criterio di ordinamento

#cfigure("images/240418-13h26m00s-screenshot.png",100%)

=== `System.IConvertible`

- Questa interfaccia fornisce metodi per convertire il valore di un'istanza di un tipo che implementa l'interfaccia in un valore equivalente di un tipo CLR

- I tipi *CLR* sono `Boolean, SByte, Byte, Int16, UInt16, Int32, UInt32, Int64, UInt64, Single, Double, Decimal, DateTime, Char,` e `String`

- Se non esiste una conversione sensata verso un tipo CLR, l'implementazione particolare del corrispondente metodo
dell'interfaccia deve lanciare
una InvalidCastException

- Ad esempio, questa interfaccia è implementata dal tipo Boolean;
    l'implementazione del metodo ToDateTimelancia un'eccezione
    in quanto non esiste alcun valore DateTimeequivalente
    a un valore del un tipo Boolean

#cfigure("images/2024-04-18-13-49-39.png",20%)

=== `System.Convert`

- In `System.Int32`, l'implementazione
    dell'interfaccia `System.IConvertible`
    è un esempio di “_explicit interface
    implementation_”:

```C
int x = 32;
double d = x.ToDouble (...); // No!
```
È necessario scrivere:

```C
((IConvertible) x).ToDouble (...)
```

- Se necessario, utilizzare la classe Convert:

```C
Convert.ToDouble(x)
```
#cfigure("images/2024-04-18-13-50-59.png",20%)

=== System.Convert

- Lancia un'eccezione se la conversione non è supportata

```C
bool b = Convert.ToBoolean(DateTime.Today);
// InvalidCastException
```
- Effettua conversioni controllate

    ```C
    int k = 300;
    byte b = (byte) k; // b == 44
    byte b = Convert.ToByte(k); // OverflowException
    ```
    - In alcuni casi, esegue un arrotondamento:
    ```C
    double d = 42.72;
    int k = (int) d; // k == 42
    int k = Convert.ToInt32(d); // k == 43
    ```
- È utile anche quando si ha una stringche deve essere convertita
    in valore numerico:
    ```C
    string myString = "123456789";
    int myInt = Convert.ToInt32(myString);```

=== Conversione di tipo

- *Una conversione di ampliamento* avviene quando un valore di un tipo viene convertito verso un altro tipo che è di dimensione uguale o superiore
    - Da `Int32` a `Int64`
    - Da `Int32` a `UInt64`
    - Da `Int32` a `Single` (con possibile perdita di precisione)
    - Da `Int32` a `Double`
- Una conversione di restrizione avviene quando un valore di un tipo viene convertito verso un altro tipo che è di dimensione inferiore
    - Da `Int32` a `Byte`
    - Da `Int32` a `SByte`
    - Da `Int32` a `Int16`
    - Da `Int32` a `UInt16`
    - Da `Int32` a `UInt32`



- *Conversioni implicite* - non generano eccezioni
    - *Conversioni numeriche*\
       Il tipo di destinazione dovrebbe essere in grado di contenere,
       senza perdita di informazione, tutti i valori ammessi dal tipo di partenza\
       Eccezione:
        ```C
        int k1 = 1234567891;
        float b = k1;
        int k2 = (int) b; // k2 == 1234567936
        ```
    - *Up cast*\
    *Principio di sostituibilità:* deve sempre essere possibile utilizzare
    una classe derivata al posto della classe base

        ```C
        B b = new B(...); // class B : A
        A a = b;
        ```


- *Conversioni esplicite* - possono generare eccezioni
    - *Conversioni numeriche*
       Il tipo di destinazione non sempre è in grado di contenere
       il valore del tipo di partenza

```C
int k1 = -1234567891;
uint k2 = (uint) k1; // k2 == 3060399405
```
```C
int k1 = -1234567891;
uint k2 = checked((uint) k1); // OverflowException
```
```C
int k1 = -1234567891;
uint k2 = Convert.ToUInt32(k1); // OverflowException
```

- *Conversioni esplicite* - possono generare eccezioni
    - *Down cast*

```C
A a = new B(...); // class B : A
B b = (B) a; // Ok
```
```C
a = new A(...);
b = (B) a; // InvalidCastException
```
```C
if(a is B) // if(a.GetType() == typeof(B))
{
    b = (B) a; // Non genera eccezioni
    ...
}
```
```C
b = a as B; // b = (a is B)? (B) a : null;
if(b != null)
{
    ...
}
```


- *Boxing - up cast* (conversione implicita)

```C
int k1 = 100;
object o = k1; // Copia!
k1 = 200;
```
- *Unboxing - down cast* (conversione esplicita)

```C
int k2 = (int) o; // k1 = 200, k2 = 100
```
```C
double d1 = (double) k1; // Ok
d1 = k1; // Ok
d1 = o; // Non compila!
d1 = (double) o; // InvalidCastException
d1 = (int) o; // Ok
```

====  Conversione di tipo definite dall'utente

```C
public static implicit operator typeOut(typeIn obj)
public static explicit operator typeOut(typeIn obj)
```
- Metodi statici di una classe o di una struttura
- La keyword `implicit` indica l'utilizzo automatico (cast implicito)

Il metodo non deve generare eccezioni

- La keyword `explicit` indica la necessità di un cast esplicito

Il metodo può generare eccezioni

- `typeOut` è il tipo del risultato del cast
- `typeIn` è il tipo del valore da convertire
- `typeIn` o `typeOut` deve essere il tipo che contiene il metodo



=== Conversioni a string

- Conversioni a `string` (di un `Int32`):
    - ```C ToString()```
       ```c
       int k1 = -1234567891;
       string str = k1.ToString(); // str == “ - 1234567891”```
    - ```c ToString(string formatString)```\
       l'istanza è formattata secondo il `NumberFormatInfo`
       dell'impostazione cultura (_culture_) corrente
       ```C
       k1.ToString(“X”); // = “B669FD2D”
       k1.ToString(“C”); // = “ - € 1.234.567.891,00”
       k1.ToString(“C0”); // = “ - € 1.234.567.891”
       k1.ToString(“N0”); // = “ - 1.234.567.891”
       k1.ToString(“E”); // = “ - 1,234568E+009”```


- Conversioni a `string` (di un `Int32`):
    - ```C String.Format(string format, params object[] args)```

Il parametro `format` è costituito da uno o più elementi di formato
nella forma: ```C {index[,alignment][:formatString]}```
```C
int k1 = -1234567891;
String.Format (“{0}”,k1); // = “ - 1234567891”
String.Format (“{0:X}”,k1); // = “B669FD2D”
String.Format (“{0,5:X}”,k1); // = “B669FD2D”
String.Format (“{0,10:X}”,k1); // = “ ΔΔ B669FD2D”
String.Format (“{0, - 10:X}”,k1); // = “B669FD2D ΔΔ ”
String.Format (“{0:N0}”,k1); // = “ - 1.234.567.891”
```


=== Conversioni da string
- Conversioni da `string` (in un `Int32`):
    - ```C Int32.Parse(string str)
       Int32.Parse(“ - 1234567891”); // - 1234567891
       Int32.Parse(“ - 1.234.567.891”); // FormatException
       Int32.Parse(“”); // FormatException
       Int32.Parse(“ - 1234567891999”); // OverflowException
       Int32.Parse(null); // ArgumentNullException```
    - ```C Int32.Parse(string str, System.Globalization.NumberStyles style)```
       `NumberStyles` determina lo stile permesso per i parametri
       stringa passati ai metodi Parsedelle classi basi numeriche


- I simboli da usare per la valuta, il separatore delle migliaia, il separatore decimale e il simbolo del segno sono specificati da `NumberFormatInfo`
- Gli attributi di NumberStylessono indicati usando l'OR (inclusivo) bit-a-bit dei vari flag di campo


```C
Int32.Parse(“ - 1.234.567.891”, System.Globalization.NumberStyles.Number); // ok
Int32.Parse(“B669FD2D”, System.Globalization.NumberStyles.HexNumber); // ok
```

#cfigure("images/2024-04-18-14-02-35.png",25%)


=== Conversioni a/da string

- Conversioni a `string` (di un `Int32`):
    - ```C Convert.ToString(int value, int toBase)
       toBase= 2, 8, 10, 16```
       ```C int k1 = -1234567891;
       Convert.ToString (k1); // “ - 1234567891”
       Convert.ToString (k1,10); // “ - 1234567891”
       Convert.ToString (k1,16); // “b669fd2d”```
- Conversioni da `string` (in un `Int32`):
    - ```C Convert.ToInt32(string str, int fromBase)
       fromBase= 2, 8, 10, 16```
       ```C Convert.ToInt32(“-1234567891”); // -1234567891
       Convert.ToInt32(“-1234567891”,10); // -1234567891
       Convert.ToInt32(“B669FD2D”,16); // -1234567891
       Convert.ToInt32(“0xB669FD2D”,16); // -1234567891
       Convert.ToInt32(“B669FD2D”,10); // FormatException```


=== `System.Double`

- Segue le specifiche IEEE 754
- Supporta ±0, ±Infinity, NaN
- Epsilon rappresenta il più piccolo
    Double positivo > 0
- Il metodo TryParse è analogo al
    metodo Parse, ma non lancia eccezioni
    in caso di fallimento della conversione
       - Se la conversione ha successo,
          il valore di ritorno è true
          e il parametro di uscita è posto
          pari al risultato della conversione
       - Se la conversione fallisce,
          il valore di ritorno è false
          e il parametro di uscita è posto
          pari a zero

#cfigure("images/2024-04-18-14-04-29.png",45%)

=== `System.Enum`

- `Enum` fornisce metodi per
    - Confrontare istanze di questa classe
    - Convertire il valore di un'istanza
       nella sua rappresentazione a stringa
    - Convertire la rappresentazione
       a stringa di un numero in un'istanza
       della classe
    - Creare un'istanza di un'enumerazione e
       valore specifico
- È possibile trattare un `Enum` come un campo bit

#cfigure("images/2024-04-18-16-02-37.png",40%)

=== `System.DateTime`

- Rappresenta un istante nel
    tempo, tipicamente espresso
    come data e ora del giorno
- Il tipo di valore `DateTime`
    rappresenta le date e le ore
    con valori che vanno dalla
    mezzanotte del 1 gennaio 0001
    d.C. (Era Comune) alle 11:59:59
    P.M., 31 dicembre 9999 d.C.
- I valori temporali sono misurati
    in unità di 100 ns chiamate tick
- `DateTime` rappresenta
    un istante nel tempo,
    mentre `TimeSpan` rappresenta
    un intervallo di tempo

#cfigure("images/2024-04-18-16-04-04.png",55%)

=== `System.String`

#cfigure("images/2024-04-18-16-05-35.png",90%)

- Una stringa immutabile di caratteri `Unicode` a lunghezza
- Una `String` è detta immutabile perché, una volta creata,
    il suo valore non può essere modificato
- I metodi che sembrano modificare una `String`,
    in realtà restituiscono una nuova `String` contenente la modifica
- Se è necessario modificare il contenuto effettivo di un oggetto tipo stringa,
    utilizzare la classe `System.Text.StringBuilder`
- Per la classe `String` l'operatore di uguaglianza `==` è ridefinito, quindi, al contrario di Java, l'operatore ritorna `true` se le due stringhe che compara hanno lo stesso contenuto lessico-grafico



=== `System.ICloneable`
#cfigure("images/2024-04-18-16-06-07.png",30%)
- Supporta la clonazione, che crea una nuova istanza di una classe con lo stesso stato di un'istanza esistente
- `Clone` crea un nuovo oggetto che è una copia dell'istanza corrente
- `Clone` può essere implementato come:
    - una *copia superficiale* (shallow), vengono duplicati solo gli oggetti
       di primo livello, non vengono create nuove istanze di alcun campo

```C
public object Clone()
{
    return MemberwiseClone();
}
```
- una *copia profonda* (deiep), tutti gli oggetti vengono duplicati
- `Clone` restituisce una nuova istanza dello stesso tipo (o eventualmente un tipo derivato) dell'oggetto corrente



====  `System.Collections.IEnumerable`

- `GetEnumerator` restituisce un enumeratore
    che può essere utilizzato per scorrere una collezione
- Espone l'enumeratore, che supporta una semplice iterazione su una collezione
#cfigure("images/2024-04-18-16-09-07.png",60%)
#cfigure("images/2024-04-18-16-12-59.png",40%)
- Gli enumeratori permettono solamente di leggere i dati della collezione
- Gli enumeratori non possono essere usati per modificare la collezione
- `Reset` riporta l'enumeratore allo stato iniziale
- `MoveNext` si sposta all'elemento successivo, restituendo
    - `true` se l'operazione ha successo
    - `false` se l'enumeratore ha oltrepassato l'ultimo elemento
- `Current` restituisce l'oggetto attualmente referenziato


====   `System.Collections.IEnumerator`

- Non deve essere implementata direttamente da una classe contenitore
- Deve essere implementata da una classe separata (eventualmente annidata nella classe contenitore) che fornisce la funzionalità di iterare sulla classe contenitore
- Tale suddivisione di responsabilità permette di utilizzare contemporaneamente più enumeratori sulla stessa classe contenitore
- La classe contenitore deve implementare l'interfaccia `IEnumerable`
- Se una classe contenitore viene modificata, tutti gli enumeratori a essa associati vengono invalidati e non possono più essere utilizzati (`InvalidOperationException`)


```C IEnumerator enumerator = enumerable.GetEnumerator();

while (enumerator.MoveNext())

{
    MyType obj = (MyType) enumerator.Current;
    ...
}
```
\
```C
foreach (MyType obj in enumerable)
{
    ...
}```


```C public class Contenitore : IEnumerable
{
    ...
    public IEnumerator GetEnumerator()
    {
        return new Enumeratore(this);
    }

    class Enumeratore : IEnumerator
    {
        public Enumeratore(Contenitore contenitore ) ...
    }
}```


=== `System.Array`

#cfigure("images/2024-04-18-16-18-17.png",80%)


- Array mono-dimensionali

    ```C
    int[] a = new int[3];
    int[] b = new int[] {3, 4, 5};
    int[] c = {3, 4, 5};
    // array of references
    SomeClass[] d = new SomeClass[10];
    // array of values (directly in the array)
    SomeStruct[] e = new SomeStruct[10];
    ```


- Array multidimensionali (frastagliati): ogni array può avere dimensione diversa, per questo non possono essere inizializzati direttamente ("in un colpo solo")
    ```C
    // array of references to other arrays
    int[][] a = new int[2][];
    // cannot be initialized directly
    a[0] = new int[] {1, 2, 3};
    a[1] = new int[] {4, 5, 6};
    ```
- Array multidimensionali (matrici)

    ```C
    // block matrix
    int[,] a = new int[2, 3];
    // can be initialized directly
    int[,] b = {{1, 2, 3}, {4, 5, 6}};
    int[,,] c = new int[2, 4, 2];
    ```


- *Frastagliati* (come in Java)

    ```C
    int[][] a = new int[2][];
    a[0] = new int[3];
    a[1] = new int[4];
    ...
    int x = a[0][1];
    ```
#cfigure("images/2024-04-18-16-20-39.png",50%)
- Matrici (come in C, più compatti ed efficienti)

    ```C
    int[,] a = new int[2, 3];
    ...
    int x = a[0, 1];
    ```
#cfigure("images/2024-04-18-16-20-57.png",45%)




== Delegati ed eventi

=== Delegati

- Sono oggetti che possono contenere il riferimento (_type safe_)
    a un metodo, tramite il quale il metodo può essere invocato
- *Oggetti funzione* (_functor_): oggetti che si comportano come una funzione (metodo)
- Simili ai puntatori a funzione del C/C++,
    ma _object-oriented_ e molto più potenti
- Utilizzo standard: funzionalità di *callback*
    - *Elaborazione asincrona*
    - *Elaborazione cooperativa* (il chiamato fornisce una parte del servizio,
       il chiamante fornisce la parte rimanente - es. qsortin C)
    - *Gestione degli eventi* (chi è interessato a un certo evento
       si registra presso il generatore dell'evento,
       specificando il metodo che gestirà l'evento)


==== C/C++: Puntatori a funzioni

```C int funX(char c);
int funY(char c);
int (g)(char c) = NULL;
...
g = cond1? funX : funY;
oppure: g = cond1? &funX : &funY;
...
... g('H') ... ≡ ... (g)('H') ...
```

==== C/C++: Array di puntatori a funzioni

```C void fun0(char s);
void fun1(char s);
void fun2(char s);
void (fun[])(char s) 
{ fun0, fun1, fun2 };
...
fun[m]("stringa di caratteri"); ≡
    (fun[m])("stringa di caratteri");
```
#cfigure("images/2024-05-16-10-25-22.png",60%)


==== C/C++: Elaborazione cooperativa

```C void qsort(void base, int num, int width,
int (compare)(void , void ));```
#cfigure("images/2024-05-16-10-27-11.png",90%)



=== Delegati

- *Dichiarazione* di un nuovo *tipo* di *delegato*
    che può contenere il riferimento a un metodo
    che ha un unico argomento intero e restituisce un intero:\
    ```C delegate int Azione(int param);```
- *Definizione* di un *delegato*:

    ```C Azione azione;```

- *Inizializzazione* di un delegato:


```C azione = new Azione(nomeMetodoStatico);
azione = new Azione(obj.nomeMetodo);
azione = nomeMetodoStatico; // C2.
azione = obj.nomeMetodo; // C2.```

- *Invocazione del metodo* referenziato dal delegato:

```C int k1 = azione(10);```



==== Delegati: Multicasting

- È possibile assegnare al delegato una *lista di metodi*
    (invocation list)
- All'atto della chiamata del delegato, i metodi vengono invocati
    - in *sequenza*
    - in *modo sincrono*
- Per aggiungere un metodo alla lista: `+=`

```C Azione azione = Fun1;
... azione(10) ... // Fun1(10)
azione += Fun2;
... azione(10) ... // Fun1(10), Fun2(10)```

- Per *togliere un metodo* dalla lista: `-=`

```C azione -= Fun1;
... azione(10) ... // Fun2(10)```



#line(length: 100%)

- Un'istanza di delegato incapsula uno o più metodi (con una lista di parametri e tipo restituito specifici), ciascuno dei quali è indicato come entità  invocabile (*callable entity*)
    - per i #text(blue)[*metodi statici*], un'entità invocabile consiste
        solo in un metodo
    - per i #text(blue)[*metodi di istanza*], un'entità invocabile consiste
        in un'istanza e un metodo su quell'istanza
- Un delegato
    - applica solo una singola firma del metodo (non un nome)
    - non conosce o non si preoccupa della classe dell'oggetto a cui fa riferimento
- Ciò rende i delegati utili per chiamate anonime (*anonymous invocation*)


#cfigure("images/2024-05-16-10-32-48.png",90%)

- Possiamo intendere un delegato come un array con due colonne
    - quella di destra contiene il metodo
    - quella di sinistra contiene il riferimento all'istanza su cui il metodo è applicato (`NULL` nel caso di metodi statici)

- L'invocazione di un'istanza del delegato la cui lista di metodi
    contiene più elementi procede richiamando ciascuno dei metodi
    nella lista, in modo sincrono, in ordine
- A ogni metodo così invocato viene passato lo stesso
    insieme di parametri fornito all'istanza del delegato
- Se tale chiamata di delegato include parametri di tipo riferimento
    - ogni invocazione di metodo avverrà con un riferimento alla stessa
       variabile
    - le modifiche a tale variabile da parte di un metodo nella lista
       saranno visibili ai metodi successivi in lista
- Se la chiamata del delegato include parametri di output
    o un valore di ritorno
       - il loro valore finale verrà dall'invocazione dell'ultimo delegato nell'elenco

#cfigure("images/2024-05-16-10-33-37.png",100%)


- In C\#, la dichiarazione di un nuovo tipo di delegato definisce automaticamente una nuova classe derivata dalla classe `System.MulticastDelegate`


```C System.Object
    System.Delegate
        System.MulticastDelegate
            Azione```

- Pertanto, sulle istanze di Azione è possibile invocare i metodi definiti a livello di classi di sistema



==== Esempio Boss-Worker

- È necessario modellare un'interazione tra due componenti
    - un *Worker* che effettua un'attività (o lavoro)
    - un *Boss* che controlla l'attività dei suoi Worker
- Ogni Worker deve notificare al proprio Boss:
    - quando il lavoro inizia
    - quando il lavoro è in esecuzione
    - quando il lavoro finisce
- Soluzioni possibili:
    + class-based callback relationship
    + interface-based callback relationship
    + pattern Observer(lista di notifiche)
    + delegate-based callback relationship
    + event-based callback relationship


==== Una Relazione di Chiamata Basata su Delegati

- Un delegato è un'entità _type-safe_ che si pone tra 1 caller e 0+ call
    target e che agisce come un'interfaccia con un solo metodo

```C interface IWorkerEvents
{
    void WorkStarted(Worker worker);
    void WorkProgressing(Worker worker);
    int WorkCompleted(Worker worker);
}

delegate void WorkStarted(Worker worker);
delegate void WorkProgressing(Worker worker);
delegate int WorkCompleted(Worker worker);```


#cfigure("images/2024-05-16-11-12-01.png",90%)


*Dai Delegati agli Eventi*\
- L'utilizzo di campi pubblici per la registrazione
    fornisce un accesso eccessivo
       - I client possono sovrascrivere altri client
          precedentemente registrati\
          ```C peter.Started = WorkStarted;```
       - I client possono invocare i chiamati\
          ```C peter.Completed(peter);```
- Fornire metodi di registrazione pubblica abbinati al campo
    del delegato è una soluzione migliore,
    ma pesante se implementata manualmente
- Il modificatore event automatizza il supporto per
    - #text(blue)[*public [un]registration*]
    - #text(blue)[*private implementation*]



#cfigure("images/2024-05-16-11-15-06.png",90%)


*Personalizzare la Registrazione a Eventi*\

- È possibile definire gestori di registrazione a eventi
- Uno dei vantaggi di scrivere metodi propri di registrazione è l'aumentato controllo
- La sintassi alternativa, analoga a una proprietà, supporta gestori di registrazione definiti dall'utente
- Consente di rendere la registrazione condizionale o diversamente personalizzata
- Sintassi di accesso lato client non modificata
- È necessario fornire spazio per i client registrati


```C class Worker
{ ...
    public event WorkProgressing Progressing
    {
        add
        {
            if(DateTime.Now.Hour < 12)
            { _progressing += value; }
            else
            { throw new InvalidOperationException
                ("Must register before noon."); }
        }
        remove
        { _progressing -= value; }
    }
    private WorkProgressing _progressing;
    ...
}```


=== Eventi

- *Evento*: “Fatto o avvenimento determinante nei confronti
    di una situazione oggettiva o soggettiva”
- In programmazione, un evento può essere scatenato
    - Dall'interazione con l'utente (click del mouse, ...)
    - dalla logica del programma
- *Event sender* - l'oggetto (o la classe) che scatena
    (raiseso triggers) l'evento (sorgente dell'evento)
- *Event receiver* - l'oggetto (o la classe) per il quale l'evento
    è determinante e che quindi desidera essere notificato
    quando l'evento si verifica (cliente)
- *Event handler* - il metodo (dell'_event receiver_)
    che viene eseguito all'atto della notifica


- Quando si verifica l'evento, il sender invia un messaggio di notifica a tutti i receiver
    - in pratica, invoca gli eventhandler di tutti i receiver
- In genere, il sender NON conosce né i _receiver_, né gli _handler_

- Il meccanismo che viene utilizzato per collegare sender e receiver/handler è il delegato (che permette *invocazioni anonime*)


==== Dichiarazione di un Evento - Convenzione .NET

- Un evento incapsula un delegato
    - è necessario dichiarare un tipo di delegato prima di poter dichiarare un evento
- Per convenzione, i delegati degli eventi in .NET hanno 2 parametri
    - la *sorgente* che ha scatenato l'evento e
    - i *dati* relativi all'evento
- Molti eventi, inclusi eventi della GUI come i click del mouse,
    non generano dati
- In tali situazioni, è sufficiente usare il delegato dell'evento
    fornito dalla libreria di classi per gli eventi senza dati,
    ```C System.EventHandler```
- Delegati di eventi personalizzati sono necessari solamente
    quando un evento genera dati




```C public delegate void EventHandler(object sender, EventArgs e);

System.Object
    System.Delegate
        System.MulticastDelegate
            System.EventHandler```

- La classe `System.EventArgs` viene utilizzata quando un evento
    non deve passare informazioni aggiuntive ai propri gestori
- Se i gestori dell'evento hanno bisogno di informazioni aggiuntive,
    è necessario derivare una classe dalla classe EventArgs ,
    aggiungere i dati necessari e utilizzare il delegate
    ```C EventHandler<TEventArgs>```



```C public event EventHandler Changed;```

- In pratica, `Changed` è un delegato, ma la _keyword_ #text(blue)[`event`] ne limita
    - la visibilità e
    - le possibilità di utilizzo
- Una volta dichiarato, l'evento può essere trattato
    come un delegato di tipo speciale
- In particolare, può:
    - essere #text(blue)[`null`] se nessun cliente si è registrato
    - essere associato a uno o più metodi da invocare


==== Invocazione di un Evento

- Per scatenare un evento è opportuno definire un metodo protetto
    virtuale `OnNomeEvento` e invocare sempre quello
    public event EventHandler Changed ;
    protected virtual void OnChanged()
    ```C {
       if ( Changed != null )
          Changed ( this , EventArgs.Empty );
    }
    ...
    OnChanged ();
    ...```
- #underline[*Limitazione rispetto ai delegati*]\
    L'invocazione dell'evento può avvenire solo
    all'interno della classe nella quale l'evento è stato dichiarato
    (benché l'evento sia stato dichiarato #text(blue)[`public`])



- Al di fuori della classe in cui l'evento è stato dichiarato,
    un evento viene visto come un *delegato con accessi molto limitati*
- Le sole operazioni effettuabili dal cliente sono:
    - #text(blue)[agganciarsi a un evento]: aggiungere un nuovo delegato
       all'evento mediante l'operatore `+=`
    - #text(blue)[sganciarsi da un evento]: rimuovere un delegato dall'evento
       mediante l'operatore `-=`


===== Agganciarsi a un Evento


Per iniziare a ricevere le notifiche di un evento, il cliente deve:

- *Definire il metodo* (_event handler_) che dovrà essere invocato
    all'atto della notifica dell'evento (con la stessa signature dell'evento):
    ```C void ListChanged(object sender, EventArgs e)
    { ... }```
- *Creare un delegato* dello stesso tipo dell'evento,
    farlo riferire al metodo
    e aggiungerlo alla lista dei delegati associati all'evento :
    ```C List.Changed += new EventHandler(ListChanged);
    List.Changed += ListChanged; // C# 2.0```


===== Sganciarsi da un Evento


Per smettere di ricevere le notifiche di un evento, il cliente deve:
- *Rimuovere il delegato* dalla lista dei delegati associati all'evento:

```C List.Changed -= new EventHandler ( ListChanged );
List.Changed -= ListChanged ; // C# 2.0```


#line(length: 100%)

- Poiché `+=` e `-=` sono gli unici operatori permessi su un evento al di
    fuori del tipo che dichiara l'evento, il codice esterno al tipo
       - può aggiungere e rimuovere handlerper un evento, ma
       - non può in nessun altro modo ottenere o modificare
          la lista di handlersottostante
- Gli eventi forniscono agli oggetti un modo utile
    per segnalare modifiche allo stato a clienti di tali oggetti
- Gli eventi sono un componente fondamentale
    *per la creazione di classi che possano essere riutilizzate
    in un gran numero di programmi differenti*














= Principi di Design



== Qualità della progettazione

- La #text(blue)[*Qualità della progettazione*] è un concetto vago
- La qualità dipende da specifiche #text(blue)[*priorità
    dell' organizazione*]
- Un buon progetto potrebbe essere
    - il più affidabile,
    - il più efficiente,
    - il più manutenibile,
    - il più economico, ...
- Gli argomenti qui discussi riguardano principalmente
    la #text(blue)[*manutenibilità del progetto*]


== Cosa rende un design “cattivo”?

#text(blue)[*✓ Misdirection*]: non soddisfa i requisiti

#text(blue)[*✓ Rigidità del software*]: una singola modifica influisce
su molte altre parti del sistema

#text(blue)[*✓ Fragilità del software*]: una singola modifica influisce
su parti inaspettate del sistema

#text(blue)[*✓ Immobilità del software*]: è difficile da riutilizzare
in un'altra applicazione

#text(blue)[*✓ Viscosità*]: è difficile fare la cosa giusta,
ma facile fare la cosa sbagliata


=== Rigidità del Software

- La tendenza per il software a essere
    #text(blue)[*difficile da modificare*]
- #text(blue)[*Sintomo*]: ogni modifica provoca una cascata
    di modifiche successive nei moduli dipendenti
- #text(blue)[*Effetto*]: i manager hanno timore ad accettare
    che gli sviluppatori risolvano problemi non critici -
    non sanno se/quando gli sviluppatori termineranno
    le modifiche


=== Fragilità del Software

- La tendenza del software a #text(blue)[*“rompersi” in molti punti*]
    ogni volta che viene modificato: i cambiamenti tendono
    a causare #text(blue)[*comportamenti inaspettati*] in altre parti
    del sistema (spesso in aree che non hanno alcuna
    relazione concettuale con l'area che è stata modificata)
- #text(blue)[*Sintomo*]: ogni correzione peggiora le cose,
    introducendo più problemi di quelli risolti: tale software
    è impossibile da manutenere
- #text(blue)[*Effetto*]: ogni volta che i manager autorizzano
    una correzione, temono che il software si “rompa”
    in modo inaspettato


=== Immobilità del Software

- L'#text(blue)[*impossibilità di riutilizzare il software*] di altri progetti
    o di parti dello stesso progetto
- #text(blue)[*Sintomo*]: uno sviluppatore scopre di aver bisogno
    di un modulo simile a quello scritto da un altro
    sviluppatore. Ma il modulo in questione ha troppe
    dipendenze. Dopo molto lavoro, lo sviluppatore scopre
    che il lavoro e il rischio necessari per separare le parti
    desiderabili del software dalle parti indesiderabili sono
    troppo grandi per essere tollerati
- #text(blue)[*Effetto*]: e così il software viene semplicemente riscritto
    anziché riutilizzato


=== Viscosità del Software

- Gli sviluppatori di solito trovano più di un modo per apportare
    una modifica
       - alcuni preservano il design, altri no (cioè sono hack)
- La tendenza a #text(blue)[*incoraggiare modifiche al software
    che sono hack*] piuttosto che modifiche al software
    che preservano l'intento di progettazione originale
       - #text(blue)[*Viscosità del design*]: i metodi che preservano il design
          sono più difficili da utilizzare rispetto agli hack
       - #text(blue)[*Viscosità dell'ambiente*]: l'ambiente di sviluppo è lento e
          inefficiente (tempi di compilazione molto lunghi, il sistema
          di controllo dei sorgenti richiede ore per archiviare pochi file, ...)
- #text(blue)[*Sintomo*]: è facile fare la cosa sbagliata,
    ma difficile fare la cosa giusta, cioè è difficile fare modifiche se seguano il design indicato nel documento
- #text(blue)[*Effetto*]: la manutenibilità del software degenera
    a causa di hack, scorciatoie, correzioni temporanee, ...


===  Perché esistono risultati di progettazione scadenti?

- Ragioni ovvie:
    - mancanza di capacità/pratica di progettazione
    - tecnologie in evoluzione
    - vincoli di tempo/risorse
    - complessità del dominio, ...
- Non così ovvie:
    - #text(blue)[*la “putrefazione” del software è un processo lento*] -
       anche un design originariamente pulito ed elegante
       può degenerare nel corso dei mesi/anni
    - i #text(blue)[*requisiti*] spesso cambiano in modi che non erano stati
       previsti dal design (o dal progettista) originale
    - le #text(blue)[*dipendenze*] tra moduli non pianificate e improprie
       si insinuano: le dipendenze non vengono gestite


== Modifiche ai Requisiti

- Come ingegneri del software, sappiamo benissimo
    che i requisiti cambiano
- In effetti, la maggior parte di noi si rende conto
    che il documento dei requisiti è il #text(blue)[*documento
    più volatile*] dell'intero progetto
- Se i nostri progetti falliscono a causa del costante arrivo
    di requisiti in continua evoluzione,
    #text(blue)[*la colpa è della nostra progettazione*]
- Dobbiamo in qualche modo trovare un modo
    per rendere i nostri progetti resistenti a tali cambiamenti
    e proteggerli dalla putrefazione


== Gestione delle Dipendenze

- Ciascuno dei quattro sintomi sopra menzionati è causato
    (direttamente o indirettamente) da #text(blue)[*dipendenze
    improprie tra moduli*] del software
- È l'#text(blue)[*architettura delle dipendenze*]
    che si sta degradando e con essa la capacità
    del software di essere manutenuto
- Per prevenire il degrado dell'architettura
    delle dipendenze, è necessario gestire le dipendenze
    tra i moduli in un'applicazione
- La progettazione orientata agli oggetti è piena di #text(blue)[*principi*]
    e #text(blue)[*tecniche*] per la gestione delle dipendenze dei moduli


== Principi di Design

- The #text(blue)[*Single Responsibility Principle*] (SRP)
- The #text(blue)[*Dependency Inversion Principle*] (DIP)
- The #text(blue)[*Interface Segregation Principle*] (ISP)
- The #text(blue)[*Open/Closed Principle*] (OCP)
- The #text(blue)[*Liskov Substitution Principle*] (LSP)


===  Premessa

==== Il principio zero

- Il principio zero è un principio di logica noto come
    #text(blue)[*rasoio di Occam*]:\
    “_Entia non sunt multiplicanda praeter necessitatem_”\
    ovvero: non bisogna introdurre concetti
    che non siano strettamente necessari
- È la forma “colta” di un principio pratico:\
    “_Quello che non c'è non si rompe_” (H. Ford)
- Tra due soluzioni bisogna preferire quella
    - che introduce il minor numero di ipotesi
    - che fa uso del minor numero di concetti



==== Semplicità e semplicismo

- La #text(blue)[*semplicità*] è un fattore importantissimo
    - il software deve fare i conti con una notevole componente
       di complessità, generata dal contesto in cui deve essere utilizzato
    - quindi è estremamente importante #text(blue)[*non aggiungere
       altra complessità*] arbitraria
- Il problema è che
    - la semplicità richiede uno #text(blue)[*sforzo non indifferente*]
       (#text(blue)[è molto più facile essere complicati che semplici])
    - in generale le soluzioni più semplici vengono in mente per ultime
- Bisogna fare poi molta attenzione a essere semplici
    ma non semplicistici\
    “_Keep it as simple as possible but not simpler_”
    (A. Einstein)


==== Divide et impera

- La #text(blue)[decomposizione] è una tecnica fondamentale
    per il controllo e la gestione della complessità
- Non esiste un solo modo per decomporre il sistema
    - la #text(blue)[*qualità della progettazione*] dipende direttamente dalla #text(blue)[*qualità delle scelte di decomposizione*] adottate
- In questo contesto il #text(blue)[*principio fondamentale*] è:
    minimizzare il grado di accoppiamento
    tra i moduli del sistema
- Da questo principio è possibile ricavare diverse regole:
    - minimizzare la quantità di interazione fra i moduli
    - eliminare tutti i riferimenti circolari fra moduli
    - ...


====  Rendere privati tutti i dati degli oggetti

- Le modifiche ai dati pubblici rischiano sempre di “aprire”
    il modulo:
       - possono avere un effetto domino che porta a richiedere
          modifiche in molte posizioni impreviste
       - gli errori possono essere difficili da trovare e correggere
          completamente - le correzioni possono provocare errori altrove

#cfigure("images/2024-04-22-17-37-29.png",100%)

===  The Single Responsibility Principle

- _There should never be more than one reason
    for a class to change_ (R. Martin)
- _A class has a single responsibility: it does it all,
    does it well, and does it only_ (1-Responsibility Rule)

- Se una classe ha più di una responsabilità,
    queste diventano accoppiate
- Le modifiche a una responsabilità
    possono compromettere o inibire la capacità della classe
    di realizzare le altre
- Questo tipo di accoppiamento porta a #text(blue)[*design fragili*]
    che si rompono in modi inaspettati quando modificati


#cfigure("images/2024-04-22-17-38-36.png",90%)


==== Esempio

#cfigure("images/2024-04-22-17-39-18.png",90%)

- Un'applicazione è di geometria computazionale
    - usa `Rectangle` per aiutarsi con la matematica delle forme geometriche
    - non disegna mai il rettangolo sullo schermo
- L'altra applicazione è di natura grafica
    - può anche fare un po' di geometria computazionale, ma
    - disegna sicuramente il rettangolo sullo schermo


- La classe `Rectangle` ha #text(blue)[*due responsabilità*]:
    - la prima responsabilità è fornire un modello matematico
       della geometria di un rettangolo
    - la seconda responsabilità è disegnare
       il rettangolo su un'interfaccia grafica


====  Esempio - Refactoring

- Un progetto migliore consiste nel #text(blue)[separare le due responsabilità
    in due classi completamente distinte]
- #text(blue)[Si estrae una classe]: si crea una nuova classe e si spostano
    i campi e i metodi opportuni dalla vecchia classe alla nuova classe

#cfigure("images/2024-04-22-17-40-54.png",80%)

===  The Dependency Inversion Principle

- #text(blue)[*Depend upon abstractions*]\
    #text(blue)[*Do not depend upon concretions*]
- Ogni dipendenza dovrebbe puntare a un'interfaccia
    o a una classe astratta
- Nessuna dipendenza dovrebbe puntare
    a una classe concreta
- I moduli di alto livello (i clienti) non dovrebbero dipendere
    dai moduli di basso livello (i fornitori di servizi)\
    Entrambi dovrebbero dipendere da astrazioni

#cfigure("images/2024-04-22-17-43-15.png",50%)

- I #text(blue)[*moduli di basso livello*] contengono la maggior parte del codice
    e della logica implementativa e quindi #text(blue)[*sono i più soggetti
    a cambiamenti*]
- Se #text(blue)[*i moduli di alto livello*] dipendono dai dettagli dei moduli
    di basso livello (sono accoppiati in modo troppo stretto),
    i cambiamenti si propagano e le conseguenze sono:
- #text(blue)[*Rigidità*]: bisogna intervenire su un numero elevato di moduli
- #text(blue)[*Fragilità*]: si introducono errori in altre parti del sistema
- #text(blue)[*Immobilità*]: i moduli di alto livello non si possono riutilizzare perché non si riescono a separare da quelli di basso livello

- Soluzione:
    - i moduli di basso livello implementano un'intefaccia
    - i moduli di alto livello utilizzano quell'interfaccia


#cfigure("images/2024-04-22-17-43-41.png",100%)

- Questo principio funziona perché:
    - #text(blue)[*le astrazioni*] contengono pochissimo codice (in teoria nulla)
       e quindi #text(blue)[*sono poco soggette a cambiamenti*]
    - i #text(blue)[*moduli non astratti sono soggetti a cambiamenti*]
       ma questi cambiamenti sono sicuri perché nessuno dipende
       da questi moduli
- I dettagli del sistema sono stati isolati, separati
    da un #text(blue)[*muro di astrazioni stabili*], e questo impedisce
    ai cambiamenti di propagarsi (#text(blue)[*design for change*])
- Nel contempo i singoli moduli sono #text(blue)[*maggiormente
    riusabili*] perché sono disaccoppiati fra di loro
    (#text(blue)[*design for reuse*])



====  Dipendenze transitive

- “_...all well structured object-oriented architectures
    have clearly-defined layers, with each layer
    providing some coherent set of services though a
    well-defined and controlled interface_” (Grady Booch)
- I sistemi software dovrebbero essere stratificati,
    cioè organizzati a livelli
- Le #text(blue)[*dipendenze transitive*] devono essere eliminate

#cfigure("images/2024-04-22-17-45-51.png",80%)


#cfigure("images/2024-04-22-17-46-39.png",90%)


====  Dipendenze cicliche

- Le #text(blue)[*dipendenze cicliche*] devono essere eliminate

#cfigure("images/2024-04-22-17-47-24.png",90%)


====  Stabilità delle astrazioni

#cfigure("images/2024-04-22-17-47-58.png",90%)

===  The Interface Segregation Principle

- #text(blue)[*Clients should not be forced to depend
    upon interfaces that they do not use*]
- Molte interfacce specifiche per un cliente sono meglio
    di un'unica interfaccia general purpose


====  Fat Interface

#cfigure("images/2024-04-22-17-55-04.png",90%)

===  The Interface Segregation Principle

- I clienti non dovrebbero dipendere da servizi che non utilizzano
- Le _fat interface_ creano una forma indiretta di accoppiamento (inutile)
    fra i clienti -se un cliente richiede l'aggiunta
    di una nuova funzionalità all'interfaccia, ogni altro cliente è costretto
    a cambiare anche se non è interessato alla nuova funzionalità
- Questo crea un'inutile sforzo di manutenzione e può rendere difficile
    trovare eventuali errori

- Se i servizi di una classe possono essere suddivisi in gruppi
    e ogni gruppo viene utilizzato da un diverso insieme di clienti,
    creare interfacce specifiche per ogni tipo di cliente
    e implementare tutte le interfacce nella classe



====  Segregated Interfaces

#cfigure("images/2024-04-22-17-58-26.png",100%)


#cfigure("images/2024-04-22-17-59-19.png",100%)

=== The Open/Closed Principle

- #text(blue)[*Il principio più importante per la progettazione
    di entità riutilizzabili*]
- _Software entities (classes, modules, functions _, ...)_
    should be open for extension, but closed for
    modification_
- #text(blue)[*Open*]:
    - #text(blue)[*Possono essere estese*] aggiungendo nuovo stato
       o proprietà comportamentali
- #text(blue)[*Closed*]:
    - Hanno un'interfaccia ben-definita, pubblica e stabile
       che #text(blue)[*non può essere cambiata*]



- Dobbiamo scrivere i moduli in modo che
    - #text(blue)[*possano essere estesi*],
    - #text(blue)[*senza*] la necessità di #text(blue)[*essere modificati*]
- In altre parole, vogliamo
    - cambiare quello che fanno i moduli
    - senza cambiare il codice dei moduli
- Apparentemente si tratta di una #text(blue)[*contraddizione*]:
    come può un modulo immutabile esibire un
    comportamento che non sia fisso nel tempo?
- La risposta risiede #text(blue)[*nell'astrazione*] : è possibile creare
    astrazioni che rendono un modulo immutabile,
    ma rappresentano un gruppo illimitato di comportamenti


- Il segreto sta nell'utilizzo di #text(blue)[*interfacce*]
    (o di #text(blue)[*classi astratte*])
- A un'interfaccia #text(blue)[*immutabile*] possono corrispondere
    innumerevoli classi concrete che realizzano
    comportamenti diversi
- Un modulo che utilizza astrazioni
    - non dovrà mai essere modificato,
       dal momento che le astrazioni sono immutabili
       (#text(blue)[*il modulo è chiuso per le modifiche*])
    - potrà cambiare comportamento, se si utilizzano nuove
       classi che implementano le astrazioni
       (#text(blue)[*il modulo è aperto per le estensioni*])



==== Esempio 1

- Consideriamo un semplice
    programma che si occupa
    di copiare su una stampante
    i caratteri digitati su una
    tastiera
- Assumiamo, inoltre,
    che la piattaforma di
    implementazione
    non possieda un sistema
    operativo in grado di
    supportare l'indipendenza
    dal dispositivo

#cfigure("images/2024-04-22-18-03-59.png",60%)

```C
void Copy()
{
    int c;
    while ((c = ReadKeyboard()) != EOF)
        WritePrinter(c);
}
```
```
Copy
```
```
ReadKeyboard WritePrinter
```


- #text(blue)[*I due moduli di basso livello
    sono riutilizzabili*]: possono
    essere usati in tanti altri programmi
    per accedere alla tastiera e alla
    stampante - è la stessa riusabilità
    offerta dalle librerie di classi
- #text(blue)[*Il modulo “Copy” non è
    riutilizzabile*] in un qualsiasi
    contesto che non includa una
    tastiera o una stampante
- È un peccato, perché tale modulo
    contiene “l'intelligenza del sistema”
    - è il modulo “Copy” che incapsula
    la funzionalità cui siamo interessati
    per il riuso


- Consideriamo un nuovo
    programma che copi caratteri
    da tastiera a un file su disco
- Potremmo modificare il modulo
    “Copy” per fornirgli questa nuova
    funzionalità
- Con l'andar del tempo, più e più
    dispositivi verranno aggiunti
    a questo programma di copia,
    e il modulo “Copy” sarà tappezzato
    di istruzioni if/else, diventando
    dipendente da diversi moduli
    di più basso livello
    - alla fine diverrà #text(blue)[*rigido*] e #text(blue)[*fragile*]

```C
enum OutputDevice
{
    Printer,
    Disk
};
void Copy(OutputDevice dev)
{
    int c;
    while ((c = ReadKeyboard()) != EOF)
    {
        if (dev == Printer)
            WritePrinter(c);
        else
            WriteDisk(c);
    }
}
```

- Un modo per caratterizzare il problema visto in precedenza è di
    notare che il modulo che contiene la politica di alto livello (Copy)
    dipende dai moduli di dettaglio e di più basso livello che controlla
    (WritePrintere ReadKeyboard)
- Se potessimo trovare un modo di rendere il modulo Copy
    indipendente dai dettagli che controlla, allora
       - potremmo #text(blue)[*riutilizzarlo*] liberamente
       - potremmo produrre altri programmi che usano questo modulo per
          #text(blue)[*copiare caratteri da un qualsiasi dispositivo di input
          a un qualsiasi dispositivo di output*]


```C
interface IReader
{
    int Read();
}

interface IWriter
{
    void Write(char);
}

void Copy(IReader r, IWriter w)
{
    int c;
    while ((c = r.Read()) != EOF)
        w.Write(c);
}```

#cfigure("images/2024-04-22-18-06-54.png",50%)


==== Esempio 2

#cfigure("images/2024-04-22-18-07-24.png",100%)

- #text(blue)[*Se dovessi creare una nuova*] shape, come `Triangle`,
    #text(blue)[dovrei modificare] `drawShape`
- In un'applicazione complessa l'istruzione `switch/case`
    verrebbe ripetuta più e più volte per ogni operazione
    possa essere effettuata su una shape
- Ancor peggio, ogni modulo che contenesse
    una tale istruzione `switch/case` statement
    #text(blue)[*manterrebbe una dipendenza da qualsiasi shape*]
    possa essere disegnata
       - Quindi, ogni volta una singola shape dovesse essere
          modificata in un qualsiasi modo, tutti i moduli dovrebbero
          essere ricompilati ed eventualmente modificati

#cfigure("images/2024-04-22-18-10-04.png",100%)



==== Esempio 3

#cfigure("images/2024-04-22-18-10-47.png",75%)

- Supponiamo di dover utilizzare un nuovo tipo di server!

#cfigure("images/2024-04-22-18-11-15.png",80%)

- `Client` è chiuso alle modifiche dell'implementazione
    di `IServer`
- `Client` è aperto all'estensione
    tramite nuove implementazioni di `IServer`
- Senza `IServer` , `Client` sarebbe aperto alle modifiche
    di `Server`


#cfigure("images/2024-04-22-18-12-06.png",90%)


==== Esempio 4

#cfigure("images/2024-04-22-18-12-33.png",90%)
- E se volessimo accendere un motore?
#cfigure("images/2024-04-22-18-12-50.png",100%)

==== Conlcusioni

- Se la maggior parte dei moduli di un'applicazione
    segue OCP, allora
       - è possibile aggiungere nuove funzionalità
          all'applicazione
             - #text(blue)[*aggiungendo nuovo codice*]
             - invece che #text(blue)[*cambiando codice funzionante*]
       - il codice che già funziona non è esposto a rotture


===  The Liskov Substitution Principle

- _Subclasses should be substitutable
    for their base classes_ (Barbara Liskov)
- _All derived classes must honor the contracts
    of their base classes_ (Design by Contract -
    Bertrand Meyer)

- Il cliente di una classe base deve continuare
    a funzionare correttamente se gli viene passato
    un sottotipo di tale classe base
- In altre parole: un cliente che usa istanze di una classe A
    deve poter usare istanze di una qualsiasi sottoclasse
    di A #text(blue)[*senza accorgersi della differenza*]



====  Example
#cfigure("images/2024-04-22-18-15-04.png",100%)
#cfigure("images/2024-04-22-18-15-25.png",100%)


- OCP si basa sull'uso di classi concrete derivate
    da un'astrazione (interfaccia o classe astratta)
- LSP costituisce una guida per creare queste classi
    concrete mediante l'ereditarietà
- La principale causa di violazioni al principio di Liskov
    è dato dalla #text(blue)[*ridefinizione di metodi virtuali*]
    nelle classi derivate:
    è qui che bisogna riporre la massima attenzione
- La chiave per evitare le violazioni al principio di Liskov
    risiede nel #text(blue)[*Design by Contract*] (B. Meyer)


== Design by Contract

- Nel Design by Contract ogni metodo ha
    - un insieme di #text(blue)[*pre-condizioni*] - requisiti minimi che devono essere
       soddisfatti dal #underline[chiamante] perché il metodo possa essere eseguito
       correttamente
    - un insieme di #text(blue)[*post-condizioni*] - requisiti che devono essere
       soddisfatti dal #underline[metodo] nel caso di esecuzione corretta
- Questi due insiemi di condizioni costituiscono
    #text(blue)[*un contratto*] tra chi invoca il metodo (cliente)
    e il metodo stesso (e quindi la classe a cui appartiene)
    - le #text(blue)[*pre-condizioni vincolano il chiamante*]
    - le #text(blue)[*post-condizioni vincolano il metodo*]
    - se il chiamante garantisce il verificarsi delle pre-condizioni,
       il metodo garantisce il verificarsi delle post-condizioni


- Quando un metodo viene ridefinito in una sottoclasse
    - le #text(blue)[*pre-condizioni*] devono essere #text(blue)[*identiche
       o meno stringenti*]
    - le #text(blue)[*post-condizioni*] devono essere #text(blue)[*identiche
       o più stringenti*]
- Questo perché un cliente che invoca il metodo conosce
    il contratto definito a livello della classe base,
    quindi non è in grado:
    - di soddisfare pre-condizioni più stringenti o
    - di accettare post-condizioni meno stringenti
- In caso contrario, il cliente dovrebbe conoscere
    informazioni sulla classe derivata e questo porterebbe
    a una violazione del principio di Liskov


```C
public class BaseClass
{
    public virtual int Calculate(int val)
    {
    Precondition(-10000 <= val && val <= 10000);
    int result = val / 100;
    Postcondition(-100 <= result && result <= 100);
    return result;
    }
}

public class SubClass : BaseClass
{
    public override int Calculate(int val)
    {
        Precondition(-20000 <= val && val <= 20000);
        int result = Math.Abs(val) / 200;
        Postcondition(0 <= result && result <= 100);
        return result;
    }
}```


== Il Quadrato è un Rettangolo?

```C public class Rectangle
{
    private double _width;
    private double _height;

    public double Width
    {
        get { return _width; }
        set { _width = value; }
    }

    public double Height
    {
        get { return _height; }
        set { _height = value; }
    }
    public double Area
    {
        get { return Width * Height; }
    }
}```

- Immaginiamo che questa
    applicazione funzioni
    correttamente e sia installata
    in diversi ambienti
- Come per tutti i software
    di successo, le necessità
    degli utenti cambiano
    e si rendono necessarie nuove
    funzionalità
- Immaginiamo che, un bel
    giorno, gli utenti chiedano
    la possibilità di manipolare
    quadrati oltre che rettangoli


#cfigure("images/2024-04-22-18-18-52.png",30%)
- L'ereditarietà è una relazione IsA
- In altre parole, perché un nuovo
    tipo di oggetto verifichi la
    relazione IsAcon un tipo di
    oggetto esistente, la classe del
    nuovo oggetto deve essere
    derivata dalla classe dell'oggetto
    esistente
- Chiaramente, un quadrato
    è un rettangolo per tutti
    gli utilizzi e intenti normali
- Poiché vale la relazione IsA,
    è logico medellare Square
    come sottoclasse di `Rectangle`


- Questo utilizzo della relazione IsA è considerato
    da molti come una delle tecniche più importanti
    dell'Analisi Object Oriented
- Un quadrato è un tipo particolare di rettangolo,
    quindi la classe `Square` deve venire derivata
    dalla classe `Rectangle`
- Questo modo di pensare, però, può portare a problemi
    sottili, ma significativi
- In genere, tali problemi non vengono scoperti
    se non nella fase di implementazione dell'applicazione



```C public class Square : Rectangle
{
    public new double Width
    {
        get { return base.Width; }
        set
        {
            base.Width = value;
            base.Height = value;
        }
    }
    public new double Height
    {
        get { return base.Height; }
        set
        {
            base.Height = value;
            base.Width = value;
        }
    }
}```

- È necessario ridefinire le
    proprietà `Width` e `Height` ...
- Notevoli differenze tra Java
    e C++/C\#
    - In Java tutti i metodi sono
       virtuali
          - a parte i metodi `final`
    - In C++ / C\# è possibile
       definire
          - sia metodi virtuali,
          - sia #text(blue)[*metodi non virtuali*]
             (non polimorfici)



```C ...
Square s = new Square();
s.Width = 5; // 5 x 5
...
... Method1(s); //?
...

void Method1(Rectangle r)
{
    r.Width = 10;
}```

- Se invochiamo `Method1` con un
    riferimento a un oggetto `Square`,
    l'oggetto `Square` non funzionerà
    correttamente in quanto l'altezza
    non verrà modificata
- Questa è una chiara violazione
    di LSP
- `Method1` non funziona
    per sottotipi dei suoi parametri
- Il motivo di questo
    malfunzionamento è che
    Width e Height non sono state
    dichiarate virtuali in Rectangle


```C public class Rectangle
{
    ...
    public virtual double Width
    { ... }
    public virtual double Height
    { ... }
    ...
}

public class Square : Rectangle
{
    public override double Width
    { ... }
    public override double Height
    { ... }
}```

- Possiamo risolvere facilmente
- In ogni modo, quando la creazione
    di una classe derivata ci obbliga
    a modificare la classe base, spesso
    significa che il design è difettoso
- Infatti, viola l'OCP
- Potremmo rispondere argomentando
    che il vero difetto di progettazione
    è stato dimenticare di rendere virtuali
    Width e Height , e solo ora
    lo stiamo risolvendo
- Tuttavia, questo è difficile da giustificare
    poiché impostare l'altezza e la larghezza
    di un rettangolo sono operazioni
    estremamente primitive -con quale
    ragionamento le avremmo dovute
    rendere virtuali se non prevedendo
    l'esistenza del quadrato?



- A questo punto abbiamo due classi, `Square` e `Rectangle`,
    che apparentemente funzionano correttamente
- Indipendentemente da ciò che facciamo con un oggetto
    `Square`, questo rimarrà coerente con un quadrato
    matematico
- E indipendentemente da ciò che facciamo con un oggetto
    `Rectangle`, questo rimarrà un rettangolo matematico
- Inoltre, possiamo passare uno `Square` a una funzione
    che accetta un riferimento a un `Rectangle` e lo `Square`
    agirà comunque come un quadrato e rimarrà consistente
- Pertanto, potremmo concludere che il modello ora
    è consistente e corretto in sé
- Tuttavia, #text(blue)[*un modello consistente in sé
    non è necessariamente consistente con tutti i suoi utenti!*]



```C
public void Scale(Rectangle rectangle)
{
    rectangle.Width = rectangle.Width * ScalingFactor;
    rectangle.Height = rectangle.Height * ScalingFactor;
}
```
- `Scale` invoca membri di ciò che crede essere
    un `Rectangle`
- Sostituendovi uno Square otterremo
    che il quadrato verrà ridimensionato due volte!
- E allora qui sta il #text(blue)[*vero problema*]:
    il programmatore che ha scritto `Scale` era giustificato
    nel presumere che la modifica della larghezza
    di un `Rectangle` lasci invariata la sua altezza?



- Chiaramente, il programmatore di `Scale` ha fatto questa ipotesi assai ragionevole
- Passare uno `Square` a funzioni i cui programmatori
    hanno fatto questa ipotesi provocherà problemi
- Pertanto, esistono funzioni che accettano riferimenti
    a oggetti `Rectangle` , ma non possono operare
    correttamente su oggetti `Square`
- Queste funzioni espongono una violazione di LSP
- L'aggiunta del sottotipo `Square` di `Rectangle`
    ha guastato queste funzioni: l'OCP è stato violato



- Cosa non va nel modello di `Square` e `Rectangle`?
    - Dopo tutto, un quadrato non è un rettangolo?
    - La relazione IsA non vale?
- No! Un quadrato sarà anche un rettangolo,
    ma un oggetto `Square` non è sicuramente
    un oggetto `Rectangle`
- Perché? Perché il #text(blue)[*comportamento di un oggetto
    `Square` non è consistente con il comportamento
    di un oggetto `Rectangle`*]
- Dal punto di vista comportamentale,
    uno `Square` non è un `Rectangle`!\
    E il software si basa proprio sul comportamento



- LSP chiarisce che #text(blue)[*in OOD la relazione IsA
    riguarda il comportamento*]
       - Non comportamento privato intrinseco,
          ma #text(blue)[*comportamento pubblico estrinseco*]
       - Comportamento da cui dipendono i clienti


- Ad esempio, l'autore di `Scale` dipendeva dal fatto che i rettangoli si comportano in modo tale che #text(blue)[*le loro altezza e larghezza possano variare
indipendentemente l'una dall'altra*]
- Tale indipendenza delle due variabili
    è un #text(blue)[*comportamento pubblico estrinseco*]
    da cui probabilmente dipenderanno altri programmatori
- Affinché LSP possa valere, e con esso OCP,
    #text(blue)[*tutti i sottotipi devono essere conformi
    al comportamento che i clienti si aspettano
    dalle classi base che utilizzano*]

- La regola per le pre-condizioni e le post-condizioni
    per i sottotipi è:\
       “quando si ridefinisce una routine, si può sostituire
          #text(blue)[*la sua pre-condizione solo con una più debole*] e #text(blue)[*la sua post-condizione solo con una più forte*]”
- In altre parole, quando si utilizza un oggetto
    attraverso l'interfaccia della sua classe base,
    #text(blue)[*l'utente conosce solo le pre-condizioni
    e le post-condizioni della classe base*]



- Pertanto, le classi derivate non devono aspettarsi
    che tali utenti obbediscano a pre-condizioni più forti
    di quelle richieste dalla classe base
    - devono accettare tutto ciò che la classe base può accettare
- Inoltre, le classi derivate devono essere conformi
    a tutte le post-condizioni della classe base
    - i loro comportamenti e output non devono violare nessuno dei vincoli stabiliti per la classe base


- Il contratto di `Rectangle`
    - Altezza e larghezza sono indipendenti,
       si può modificarne una mantenendo costante l'altra
- `Square` viola il contratto della classe base


== Il Quadrato è un Rettangolo?

- Guardando al codice di test della classe `Rectangle`
    possiamo avere qualche idea del contratto di `Rectangle`:

```C [TestFixture]
public class RectangleFixture
{
    [Test]
    public void SetHeightAndWidth()
    {
        Rectangle rectangle = new Rectangle();
        int expectedWidth = 3, expectedHeight = 7;
        rectangle.Width = expectedWidth;
        rectangle.Height = expectedHeight;
        Assertion.AssertEquals(expectedWidth, rectangle.Width);
        Assertion.AssertEquals(expectedHeight, rectangle.Height);
    }
}```



```C [TestFixture] public class RectangleFixture
{
    [Test] public void SetHeightAndWidth()
    {
        Rectangle rectangle = GetShape();
        ...
    }
    protected virtual Rectangle GetShape()
    { return new Rectangle(); }
}

[TestFixture] public class SquareFixture : RectangleFixture
{
    protected override Rectangle GetShape()
    { return new Square(); }
}```


==  Principi di Architettura dei Package

- #text(blue)[*Reuse/Release Equivalency Principle*] (REP)
- #text(blue)[*Common Closure Principle*] (CCP)
- #text(blue)[*Common Reuse Principle*] (CRP)


===  Release/Reuse Equivalency Principle

- _The granule of reuse is the granule of release_


- Un elemento riutilizzabile, sia esso un componente,
    una classe o un insieme di classi, non può essere riutilizzato
    a meno che non sia gestito da un sistema di rilascio di qualche tipo
- I clienti dovrebbero rifiutare di riutilizzare un elemento a meno che
    l'autore non prometta di tenere traccia dei numeri di versione
    e di mantenere le vecchie versioni per qualche tempo
    - un criterio per raggruppare le classi in package è il riutilizzo
- Poiché i pacchetti sono l'unità di rilascio in Java,
    sono anche l'unità di riutilizzo
- Pertanto, gli architetti farebbero bene a raggruppare in package
    le classi riutilizzabili assieme


===  Common Closure Principle

- _Classes that change together, belong together_


- Il lavoro per gestire, testare e rilasciare un pacchetto
    in un sistema di grandi dimensioni non è banale
- Più pacchetti cambiano in un dato rilascio, maggiore
    è il lavoro per ricostruire, testare e distribuire il rilascio
    - vorremmo #text(blue)[*ridurre al minimo il numero di pacchetti
    che vengono modificati in un dato ciclo di rilascio
    del prodotto*]
- Per raggiungere questo obiettivo, raggruppiamo assieme
    classi che pensiamo cambieranno insieme


===  Common Reuse Principle

- _Classes that aren't reused together should not be grouped together_


- Una dipendenza da un package è una dipendenza
    da tutto ciò che è contenuto nel package
- Quando un package cambia e il suo numero di rilascio
    viene aggiornato, tutti i client di quel package
    devono verificare di funzionare con il nuovo package -
    anche se nulla di ciò che hanno usato all'interno
    del package è effettivamente cambiato
- Pertanto, le classi che non vengono riutilizzate insieme
    non dovrebbero essere raggruppate insieme



===  Discussione

- Questi tre principi non possono essere soddisfatti
    contemporaneamente
- REP e CRP semplificano la vita ai riutilizzatori,
    mentre CCP semplifica la vita ai manutentori
- CCP cerca di rendere i package più grandi possibile
    (dopotutto, se tutte le classi stanno in un solo package,
    allora solo quel package cambierà)
- CRP, tuttavia, cerca di creare package molto piccoli
- All'inizio di un progetto, gli architetti possono impostare
    la struttura dei package in modo tale che CCP domini
    per facilità di sviluppo e manutenzione
- Successivamente, quando l'architettura si stabilizza,
    gli architetti possono ri-fattorizzare la struttura dei package
    per massimizzare REP e CRP per i riutilizzatori esterni


==  Relazioni tra i Package

- #text(blue)[*Acyclic Dependencies Principle*] (ADP)
- #text(blue)[*Stable Dependencies Principle*] (SDP)
- #text(blue)[*Stable Abstractions Principle*] (SAP)


===  Acyclic Dependencies Principle

- _The dependencies between packages must not form cycles_


- Una volta apportate le modifiche a un package, gli sviluppatori possono rilasciare i package al resto del progetto
       - Prima di poter eseguire questo rilascio, tuttavia, devono verificare
          che il package funzioni
       - Per farlo, devono compilarlo e collegarlo a tutti i package da cui dipende
- Una singola dipendenza ciclica che sfugge al controllo può rendere
    l'elenco delle dipendenze molto lungo
- Quindi, qualcuno deve osservare la struttura delle dipendenze
    dei package con regolarità e interrompere i cicli ovunque compaiano



====  Esempio: Grafo dei Package Aciclico

#cfigure("images/2024-04-22-18-48-39.png",70%)


====  Esempio: Grafo dei Package Ciclico

#cfigure("images/2024-04-22-18-49-05.png",70%)


====  Discussione

- Nello scenario aciclico per rilasciare il package protocol,
    gli ingegneri dovrebbero compilarlo con l'ultima versione
    del package comm_error ed eseguire i loro test
- Nello scenario ciclico per rilasciare il package protocol,
    gli ingegneri dovrebbero compilarlo con l'ultima versione
    di comm_error, gui, comm, process, modem, file ed
    eseguire i loro test
- Come rompere un ciclo:
    - Inframezzare un nuovo package
    - Aggiungere una nuova interfaccia



====  Rompere il Ciclo Introducendo un'Interfaccia

#cfigure("images/2024-04-22-18-52-30.png",80%)

===  Stable Dependencies Principle

- _The dependencies between packages in a design
    should be in the direction of the stability
    of the packages_.\
    _A package should only depend upon packages
    that are more stable than it is_.


- I design non possono essere completamente statici
    - Una certa volatilità è necessaria se il progetto deve essere mantenuto
- Raggiungiamo questo obiettivo conformandoci al CCP
- Alcuni package sono progettati per essere volatili,
    ci aspettiamo che cambino
- Un pacchetto con molte dipendenze in entrata è molto stabile
    perché richiede molto lavoro per riconciliare qualsiasi modifica
    con tutti i pacchetti dipendenti



====  Esempio

#cfigure("images/2024-04-22-18-53-44.png",90%)

===  Stable Abstractions Principle

- _Stable packages should be abstract packages_


- La stabilità è relativa alla quantità di lavoro richiesta
    per apportare una modifica
- Un pacchetto con molte dipendenze in entrata
    è molto stabile perché richiede molto lavoro
    per riconciliare le modifiche con tutti i pacchetti
    dipendenti


====  Esempio

#cfigure("images/2024-04-22-18-55-31.png",90%)

===  Stable Dependencies/Abstractions Principles

====  Discussione

- I package in alto sono instabili e flessibili
- I package in basso sono molto difficili da modificare, perché ogni modifica a un package ne provoca altre
- I package altamente stabili nella parte inferiore del grafo
    delle dipendenze possono essere molto difficili
    da modificare, ma secondo l'OCP non devono essere
    difficili da estendere
- Se anche i pacchetti stabili in fondo sono molto astratti,
    possono essere facilmente estesi
- È possibile creare la nostra applicazione a partire
    da package instabili facili da modificare
    e package stabili facili da estendere



= Sicurezza


== Progettazione per la Sicurezza

- #text(blue)[*La sicurezza non è qualcosa
    che può essere aggiunto al sistema*]
- La sicurezza #text(red)[*deve essere progettata insieme
    al sistema*] prima dell'implementazione
- “Sicurezza” #text(blue)[*è anche un problema implementativo*]:
    spesso le vulnerabilità sono introdotte
    durante la fase di implementazione
       - È possibile ottenere un'implementazione non sicura
          da una progettazione sicura
       - Non è possibile ottenere un'implementazione sicura
          partendo da una progettazione non sicura


== Progettazione Architetturale

- La scelta dell'architettura del sistema
    influenza profondamente la sicurezza
- Un'architettura inappropriata non garantisce
    - riservatezza ed integrità delle informazioni
    - il livello di disponibilità richiesto
- Due problemi fondamentali vanno considerati
    quando si progetta l'architettura del sistema:
       - _Protezione_: come dovrebbe essere organizzato il sistema
          in modo che i beni critici possano essere protetti
          dagli attacchi esterni?
       - _Distribuzione_: come dovrebbero essere distribuiti i beni
          in modo da minimizzare gli effetti di un attacco
          andato a buon fine?



- I due problemi sono potenzialmente in conflitto:
    - se si mettono tutti i beni in un unico posto si può costruire
       un buon livello di protezione a un costo non eccessivo
    - se però la protezione fallisce tutti i beni sono compromessi
    - distribuire i beni porta ad un maggiore costo
       per la protezione
    - ci sono più possibilità che la protezione possa fallire
       se i beni sono distribuiti, ma se questo avviene
       vi è la perdita solo parziale dei beni



- Tipicamente la migliore architettura per fornire
    un alto grado di protezione è quella a layer
- I beni critici da proteggere sono posizionati
    al livello più basso
- Il numero di layer necessari varia da applicazione
    ad applicazione e dipende dalla criticità dei beni
    che devono essere protetti
- Per migliorare la protezione inoltre sarebbe bene
    che le credenziali di accesso ai diversi livelli
    fossero diverse tra loro
       - Esempio: se si adotta un meccanismo di accesso
          con password, ogni livello deve avere
          una propria password diversa da quelle degli altri livelli


=== Esempio

#cfigure("images/2024-04-23-14-12-12.png",100%)


- Se la protezione dei dati è un requisito critico
    si potrebbe anche usare un'architettura client-server
    con i meccanismi di protezione nella macchina server



- La versione tradizionale client-server ha molte limitazioni
- Se la sicurezza viene compromessa
    - le perdite associate ad un attacco saranno alte
       Esempio: tutte le credenziali di accesso
       verranno compromesse
    - i costi di recupero saranno anch'essi elevati
       Esempio: tutte le credenziali di accesso al sistema
       andranno rigenerate
- Il sistema è inoltre maggiormente soggetto ad attacchi
    DoS che sovraccaricano il server
- Una possibile soluzione può essere quella
    di adottare una architettura distribuita
    in cui il server viene replicato in punti diversi della rete


=== Esempio

#cfigure("images/2024-04-23-14-13-01.png",100%)

=== Esempio

- I beni del sistema sono distribuiti in diversi nodi
    della rete ognuno dei quali ha un proprio meccanismo
    di protezione dei dati
- I dati “più importanti” sono replicati nei diversi nodi
- Attacco ad uno specifico nodo:
    - alcuni beni non saranno disponibili
    - il sistema comunque potrà ancora funzionare
       e fornire i servizi più importanti
    - grazie alla replicazione, il ripristino dei dati
       nel nodo attaccato sarà più facile e meno costoso


#line(length: 100%)
- Tipico problema: lo stile architetturale più appropriato
    per la sicurezza potrebbe essere in conflitto
    con gli altri requisiti dell'applicazione (analisi trade-off)
- Esempio:


a. riservatezza dei dati memorizzati su un vasto database
b. accesso molto veloce ai dati

- Soddisfare entrambi i requisiti nella stessa architettura
    presenta molti problemi
       / a.: 
        - layer per garantire la riservatezza
        - diminuzione netta della velocità di accesso ai dati
       / b.: 
            - architettura “snella”
            - diminuzione netta della riservatezza



- Tipico problema: lo stile architetturale più appropriato
    per la sicurezza potrebbe essere in conflitto
    con gli altri requisiti dell'applicazione (analisi trade-off)
- Va valutato attentamente
    quale «requisito» è prioritario
    e l'architettura sarà scelta
    di conseguenza

== Linee Guida di Progettazione

- Non ci sono regole rigide per ottenere un sistema sicuro
- #text(blue)[*Differenti tipi di sistema richiedono differenti misure
    tecniche per ottenere un livello di sicurezza accettabile*]
- La posizione e i requisiti di diversi gruppi di utenti
    influenzano pesantemente #text(blue)[*cosa è*] e #text(blue)[*cosa non è*]
    accettabile
- Ci sono comunque linee guida generali
    di ampia applicabilità per la progettazione di sistemi sicuri
    che possono fungere da:
       - mezzo per migliorare la consapevolezza dei problemi
          di sicurezza in un team di progettisti software
       - base per una lista di controlli da fare durante il processo
          di validazione del sistema



+ Basare le decisioni della sicurezza su una politica esplicita
+ Evitare un singolo punto si fallimento
+ Fallire in modo certo
+ Bilanciare sicurezza e usabilità
+ Essere consapevoli dell'esistenza dell'ingegneria sociale
+ Usare ridondanza e diversità riduce i rischi
+ Validare tutti gli input
+ Dividere in compartimenti i beni
+ Progettare per il deployment
+ Progettare per il ripristino


=== Basare la Sicurezza su Policy

- La #text(blue)[*Security Policy*] è un documento di alto livello
    che definisce “cosa” è la sicurezza
    ma non “come” ottenerla
- La policy non dovrebbe definire i meccanismi usati
    per fornire e far rispettare la sicurezza
- Gli aspetti della security policy dovrebbero originare
    dei requisiti di sistema
- In pratica ciò è poco probabile, specie se viene adottato
    un processo di sviluppo rapido
- I progettisti dovrebbero quindi consultare la policy
    sia nelle decisioni di progettazione
    che nella loro valutazione


==== Progettazione delle Politiche

- Le politiche di sicurezza devono essere incorporate
    nella progettazione al fine di:
       - specificare come le informazioni possono essere accedute
       - quali precondizioni devono essere testate per l'accesso
       - a chi concedere l'accesso
- Tipicamente le politiche vengono rappresentate
    come un insieme di regole e condizioni
- Tali regole devono essere incorporate in uno specifico
    componente del sistema chiamato “#text(blue)[*_Security Authority_*]”
    che avrà il compito di far rispettare le politiche all'interno
    dell'applicazione


- A livello progettuale le politiche di sicurezza
    sono suddivise in sei specifiche categorie:
    - #highlight(fill: myred)[Identity policies:] definiscono le regole per la verifica    delle credenziali degli utenti
    - #highlight(fill: myred)[Access control policies:] definiscono le regole da applicare sia alle richieste di accesso alle risorse sia all'esecuzione di specifiche operazioni messe a disposizione dall'applicazione
    - #highlight(fill: myred)[Content-specific policies:] definiscono le regole da applicare a specifiche informazioni durante la memorizzazione e la comunicazione
    - #highlight(fill: myred)[Network and infrastructure policies:] definiscono le regole per controllare il flusso dei dati e il deployment sia delle reti  che dei servizi infrastrutturali di hosting pubblici e privati
    - #highlight(fill: myred)[Regulatory policies:] definiscono le regole a cui    l'applicazione deve sottostare per soddisfare i requisiti legali    e di regolamentazione delle leggi in vigore nel Paese/Stato  in cui il sistema opera
    - #highlight(fill: myred)[Advisor and information policies:] queste regole    non sono imposte, ma sono caldamente consigliate  in riferimento alle regole dell'organizzazione e al ruolo delle attività di business. Per esempio queste regole possono essere applicate per informare il personale sull'accesso ai dati sensibili o per stabilire comunicazioni commerciali con partner esterni


=== Evitare Punto Singolo di Fallimento

- Nei sistemi critici è buona norma di progettazione quella
    di cercare di #text(blue)[*evitare un singolo punto di fallimento*]
- Questo perché un singolo fallimento in una parte del
    sistema non si trasformi nel fallimento di tutto il sistema
- Per quanto riguarda la sicurezza questo significa
    che #text(blue)[*non ci si dovrebbe affidare a un singolo
    meccanismo per assicurarla*], ma si dovrebbero
    impiegare differenti tecniche
- Questo viene spesso chiamato “#text(red)[*difesa in profondità*]”
- Esempio: se si usa la password per autenticare
    si dovrebbe anche includere un meccanismo
    sfida e risposta


=== Fallire in Modo Certo

- Qualche tipo di fallimento è inevitabile in tutti i sistemi,
    ma i sistemi critici per la sicurezza dovrebbero sempre
    #text(red)[*_fallire in modo sicuro_*]
- Non si dovrebbero avere procedure di fall-back
    meno sicure del sistema stesso
- Anche se il sistema fallisce #text(blue)[*non deve essere
    consentito a un attaccante di accedere ai dati
    riservati*]
- Esempio: i dati dei pazienti dovrebbero essere scaricati
    sul client all'inizio di ogni sessione clinica, così se il
    server fallisce i dati sono comunque mantenuti sul client.
    I dati vengono cifrati per non essere letti da personale
    non autorizzato. Questa pratica non vale per username e password, non è necessario inserire la password nel modello del dominio.


=== Bilanciare Sicurezza e Usabilità

- Sicurezza e usabilità sono spesso in contrasto
    - #text(blue)[per avere sicurezza bisogna introdurre un numero
       di controlli] che garantiscano che gli utenti siano autorizzati
       a usare il sistema e che nello stesso tempo agiscano
       in accordo alle politiche di sicurezza
    - questo inevitabilmente ricade sull'utente che ha bisogno
       di più tempo per imparare ad utilizzare il sistema
- #text(red)[*Ogni volta che si aggiunge una caratteristica di sicurezza
    al sistema questo inevitabilmente diventa meno usabile*]
- A volte può diventare contro produttivo introdurre
    nuove caratteristiche di sicurezza a spese dell'usabilità
       - Esempio: obbligare l'utente all'adozione di password forti


=== Essere Consapevoli dell'Ingegneria Sociale

- #text(red)[*Ingegneria sociale*]: trovare modi per convincere
    con l'inganno utenti accreditati al sistema a rivelare
    informazioni riservate
- Questi approcci si avvantaggiano della “#text(blue)[*volontà di aiutare*]”
    delle persone e della loro fiducia nell'organizzazione
- Dal punto di vista della progettazione contrastare
    l'ingegneria sociale #text(blue)[*è quasi impossibile*]
- Se la sicurezza è molto critica non si dovrebbe affidarsi
    solo a meccanismi di autenticazione basati su password, ma
    bisognerebbe utilizzare meccanismi di autenticazione forte
- Meccanismi di log che tracciano sia la locazione
    che l'identità dell'utente e programmi di analisi del log
    potrebbero essere utili ad identificare brecce nella sicurezza


=== Usare Ridondanza e Diversità

- Ridondanza significa #text(blue)[*mantenere più di una versione*]
    del software e dei dati nel sistema
- Diversità significa #text(blue)[*che le diverse versioni del sistema
    non dovrebbero usare la stessa piattaforma
    o essere basati sulle stesse tecnologie*]
- In questo modo una vulnerabilità della piattaforma
    o della tecnologia non influirà su tutte le versioni
    e non condurrà a un comune punto di fallimento
- Esempio:
    - mantenere i dati dei pazienti sia sul client che sul server
    - client e server devono avere un diverso sistema operativo
    - attacco basato su vulnerabilità del SO non influenza
       sia client che server


=== Validare tutti gli Input

- Un comune attacco al sistema consiste nel fornire input
    inaspettati che causano un comportamento imprevisto
       - crash
       - perdita della disponibilità del servizio
       - esecuzione di codice malevolo
- Tipici esempi sono buffer overflow e SQL injection
- Si possono evitare molti di questi problemi progettando
    la validazione dell'input in tutto il sistema
- Nei requisiti dovrebbero essere definiti tutti i controlli
    che devono essere applicati
- Bisogna usare la conoscenza dell'input per definire
    questi controlli


=== Dividere in Compartimenti i Beni

- #text(blue)[*Compartimentalizzare*] significa organizzare
    le informazioni nel sistema in modo che gli #text(red)[*utenti
    abbiano accesso solo alle informazioni necessarie*]
    piuttosto che a tutte le informazioni del sistema
- Gli effetti di un attacco in questo modo sono più
    contenuti: qualche informazione sarà persa o
    danneggiata, ma è poco probabile che
    tutte le informazioni del sistema siano coinvolte
- Esempio:
    - lo staff clinico può avere accesso soltanto ai record
       dei pazienti che hanno un appuntamento
       o sono ricoverati nella clinica
    - esiste un meccanismo per gestire accessi inaspettati


=== Progettazione per il Deployment

- Molti problemi di sicurezza sorgono perché il sistema
    #text(red)[*non viene configurato correttamente*]
    al momento del deployment
- Bisogna sempre progettare il sistema in modo che
    - siano inclusi programmi di utilità per semplificare
       il deployment
    - verificare potenziali errori di configurazione e omissioni
       nel sistema di deployment


=== Progettazione per il Ripristino

- Bisogna sempre progettare il sistema con l'assunzione
    che gli errori di sicurezza possano accadere
- Si deve quindi pensare a come ripristinare il sistema
    dopo possibili errori e riportarlo a uno stato operazionale
    sicuro
- Esempio:
    - persone non autorizzate accedono ai dati dei pazienti
    - non è noto come abbiano ottenuto le credenziali
       per l'accesso
    - occorre quindi cambiare tutte le credenziali del sistema
       in modo che la persona non autorizzata non abbia accesso
       al meccanismo di cambiamento delle password


== Progettazione per il Deployment

- Il deployment di un sistema coinvolge:
    - configurazione del sistema per operare nell'ambiente:
       - semplice impostazione di parametri delle preferenze
          degli utenti
       - definizione di regole e modelli di business che governano
          l'esecuzione del software
    - installazione del sistema sui computer dell'ambiente
    - configurazione del sistema installato


=== Deployment del Software

#cfigure("images/2024-04-23-14-26-30.png",100%)

#line(length: 100%)

- Nella fase di deployment vengono spesso introdotte
    in modo accidentale delle vulnerabilità
- Esempio:
    - il software deve spesso essere configurato
       con una lista di utenti autorizzati
    - quando il software è rilasciato questa lista consiste
       di un login per l'amministratore generico come “admin”
       e la password di default è “password”
    - come prima azione l'amministratore dovrebbe modificare
       i dati di login, ma è molto facile dimenticare di farlo
    - un attaccante che conosce il login di default
       potrebbe essere capace di guadagnare privilegi di accesso
       al sistema



- La configurazione e il deployment sono spesso visti
    solo come problemi di amministrazione
    e quindi al di fuori del processo di ingegnerizzazione
- I progettisti software hanno la responsabilità
    di progettare per il deployment
- Bisogna sempre fornire supporti per il deployment
    che riducano la probabilità che gli amministratori
    compiano degli errori quando configurano il software
- Esistono delle linee guida per la progettazione
    per il deployment


=== Linee Guida

1. Includere supporto per visionare
    e analizzare le configurazioni
2. Minimizzare i privilegi di default
3. Localizzare le impostazioni di configurazione
4. Fornire modi per rimediare a vulnerabilità di sicurezza


==== Supporto per le Configurazioni

- Si devono sempre includere #text(blue)[*programmi di utilità*]
    che consentano agli amministratori di esaminare
    la configurazione corrente del sistema
- Sorprendentemente questi programmi #text(blue)[*mancano*]
    nella maggior parte dei sistemi software
- Gli utenti sono spesso frustrati dalla difficoltà di trovare
    i dettagli della configurazione
       - per un quadro completo della configurazione
          spesso occorre visionare diversi menu
          e questo porta a errori e omissioni
- Idealmente in fase di visualizzazione delle configurazioni
    si dovrebbero evidenziare impostazioni critiche
    per la sicurezza


==== Minimizzare i Privilegi di Default

- Il software deve essere progettato in modo tale
    che la #text(red)[*configurazione di default fornisca
    i minimi privilegi essenziali*]
- In questo modo vengono limitati i danni
    di un possibile attacco
- Esempio:
    - l'autenticazione di default dell'amministratore dovrebbe
       solo consentire l'accesso a un modulo che permette
       all'amministratore di inserire nuove credenziali
    - non dovrebbe essere consentito l'accesso
       a nessuna altra funzionalità
    - quando vengono modificate le credenziali, quella di default
       dovrebbe essere automaticamente cancellata


==== Localizzare le Impostazioni di Configurazione

- Quando si progetta il supporto per le configurazioni
    del sistema bisognerebbe assicurarsi che ogni risorsa
    che appartiene alla stessa parte del sistema
    venga configurata nella stessa posizione
- Se le informazioni di configurazione non sono localizzate
    - è facile dimenticarsi di farlo
    - può capitare di non essere a conoscenza dell'esistenza
       di meccanismi per la sicurezza già inclusi nel sistema
    - se tali meccanismi presentano configurazioni di default
       si potrebbe essere esposti ad attacchi


==== Rimediare a Vulnerabilità

- Bisogna includere #text(red)[*meccanismi diretti*] per:
    - aggiornare il sistema
    - riparare le vulnerabilità di sicurezza che vengono scoperte
- Questi potrebbero includere
    - verifiche automatiche per aggiornamenti di sicurezza
    - download di tali aggiornamenti non appena sono disponibili
- Va comunque considerato che gli aggiornamenti devono
    coinvolgere contemporaneamente centinaia di PC
    su cui tipicamente il software è installato



== Testare la Sicurezza

- Il test di un sistema gioca #text(blue)[*un ruolo chiave*] nel processo
    di sviluppo software e dovrebbe essere eseguito
    con molta attenzione
- È quindi sorprendente che #text(blue)[*l'area dei test
    della sicurezza sia quella più trascurata
    durante lo sviluppo del sistema*]
- Questo può essere attribuito a diversi fattori:
    - mancanza di comprensione dell'importanza
       dei test relativi alla sicurezza
    - mancanza di tempo
    - mancanza di conoscenza su come svolgere
       un test di sicurezza
    - mancanza di tool integrati per compiere test


- Il test della sicurezza è un lavoro molto lungo e tedioso,
    spesso molto più complesso dei test funzionali
    che vengono svolti normalmente
- Inoltre esso coinvolge diverse discipline
    - ci sono tradizionali test per accertare la sicurezza
       dei requisiti applicativi che possono essere svolti
       normalmente dal team di testing
    - ma esistono dei test non funzionali di “rottura”
       del sistema che devono essere condotti da esperti
       di sicurezza
       - Black box testing
       - White box testing


=== Black Box Testing

- Questo test ha come assunzione di base
    la non conoscenza dell'applicazione
- I tester affrontano l'applicazione
    come farebbe un attaccante
       - indagando sulle informazioni riguardanti la struttura interna
       - successivamente applicano un insieme di tentativi
          di violazione del sistema basati sulle informazioni ottenute
- Esempio: se un URL di una applicazione contiene
    una estensione “.cgi” allora può essere inferito
    che l'applicazione è stata sviluppata con la tecnologia
    CGI e applicare quindi le ben conosciute tecniche
    di violazione di questa tecnologia



- I tester possono impiegare una varietà di strumenti
    per scansionare e indagare l'applicazione
       - ci sono centinaia di tool in rete per l'hacking di applicazioni
          che permettono di “scandagliare” le porte dei sistemi
          perpetuando attacchi sfruttando le debolezze
          ben conosciute di svariati linguaggi di programmazione
- Questo test non prende in esame solo debolezze
    del codice, ma vengono svolti test mirati anche al livello
    infrastrutturale
       - errori di configurazione di reti e host
       - falle di sicurezza nelle macchine virtuali
       - problemi legati ai linguaggi di implementazione


=== White Box Testing

- Questo test ha come assunzione di base
    la completa conoscenza dell'applicazione
- I tester hanno accesso a tutte le informazioni
    di configurazione e anche al codice sorgente
- Essi operano una revisione del codice
    cercando possibili debolezze
- Inoltre scrivono test per stabilire come trarre vantaggio
    dalle debolezze scoperte
- Tipicamente questi tester sono ex-sviluppatori o persone
    che conoscono molto bene l'ambiente di sviluppo



- I tool a disposizione differiscono molto
    da quelli usati nel black box test
- Tipicamente sono tool di debugging che consentono
    di trovare bachi e vulnerabilità specifici del sistema
- I bachi tipici riguardano problemi di corsa critici
    e la mancanza di verifica dei parametri di input
    e sono specifici di ogni applicazione
- Questi test portano a scoprire anche altri problemi
    come i memory leak e problemi di prestazione
    che contribuiscono al danneggiamento della disponibilità
    e dell'affidabilità dell'intero sistema




== Capacità di Sopravvivenza del Sistema
TUTO QUESTO CAPITOLO NON È DA STUDIARE
- Con il termine #text(blue)[*capacità di sopravvivenza*] (_survivability_)
    si intende la capacità del sistema di continuare
    a fornire i servizi essenziali agli utenti legittimi
       - mentre è sotto attacco
       - dopo che parti del sistema sono state danneggiate
          come conseguenza di un attacco o di un fallimento
- La capacità di sopravvivenza è una proprietà
    dell'intero sistema, non dei singoli componenti di questo
- Il lavoro sulla capacità di sopravvivenza è molto critico
    poiché l'economia e la vita sociale dipendono
    da infrastrutture controllate da computer


- L'analisi e la progettazione della capacità
    di sopravvivenza dovrebbero essere parte del processo
    di ingegnerizzazione dei sistemi sicuri
- #text(blue)[La disponibilità dei servizi critici è l'essenza
    della sopravvivenza]
- Questo significa conoscere
    - quali sono i servizi #text(blue)[maggiormente critici]
    - come questi servizi possono essere compromessi
    - qual è la #text(blue)[qualità minima dei servizi] che deve essere
       mantenuta
    - come proteggere questi servizi
    - come #text(blue)[ripristinare velocemente] il sistema
       se i servizi diventano non disponibili


=== Esempio

- Un sistema informatico che gestisce l'invio
    delle ambulanze in risposta alle chiamate di emergenza
- Servizi:
    - prendere le chiamate e inviare le ambulanze
    - log delle chiamate
    - gestione locazione delle ambulanze
- Il servizio critico è quello legato a prendere le chiamate
    e inviare le ambulanze perché necessita di un processo
    real-time per la gestione degli eventi


=== Analisi della Sopravvivenza

- Il Survivable Analysis Systems è un metodo di analisi
    ideato agli inizi del 2000 per:
       - valutare le vulnerabilità nel sistema
       - supportare la progettazione di architetture e caratteristiche
          che promuovono la sopravvivenza del sistema
- In questo metodo la sopravvivenza del sistema
    - dipende da tre strategie complementari
    - è un processo a 4 fasi


=== Strategie

- #highlight(fill: myblue)[Resistenza:]
    - evitare problemi costruendo all'interno del sistema
       le capacità di respingere attacchi
    - es: firma digitale per l'autenticazione
- #highlight(fill: myblue)[Identificazione:]
    - individuare problemi costruendo all'interno del sistema
       le capacità di riconoscere attacchi e fallimenti
       e valutare il danno risultante
    - es: aggiungere checksum ai dati critici
- #highlight(fill: myblue)[Ripristino:]
    - tollerare problemi costruendo all'interno del sistema
       le capacità di fornire servizi essenziali durante un attacco
    - ripristinare le complete funzionalità dopo l'attacco


=== Fasi Analisi di Sopravvivenza

#cfigure("images/2024-04-24-16-59-07.png",100%)

=== Principali Attività

- #highlight(fill: myblue)[Capire il sistema:] riesaminare gli obiettivi del sistema,
    i requisiti e l'architettura
- #highlight(fill: myblue)[Identificare servizi critici:] identificare i servizi
    che devono essere mantenuti e i componenti
    che devono svolgere tale compito
- #highlight(fill: myblue)[Simulare gli attacchi:] identificare gli scenari o i casi d'uso
    dei possibili attacchi insieme ai componenti influenzati
    da questi attacchi
- #highlight(fill: myblue)[Analizzare la sopravvivenza:] identificare
    - i componenti che sono sia essenziali che a rischio
    - le strategie di sopravvivenza basate su resistenza,
       identificazione e ripristino


=== Esempio

#cfigure("images/2024-04-24-17-00-19.png",100%)

=== Esempio

- Sistema che gestisce l'equità dei prezzi nei mercati
    internazionali
- Qual è il minimo supporto che il sistema fornisce già
    per la sopravvivenza?
       - gli account dei clienti e i prezzi internazionali
          sono replicati in tutti i nodi
- Servizio chiave che deve sempre essere mantenuto:
    capacità di piazzare ordini
       - mantenere l'integrità dei dati
       - ordini accurati e che riflettano le vendite e gli acquisti
          degli utenti


=== Esempio

- Per mantenere questo servizio ci sono tre componenti
    del sistema
       - _autenticazione dell'utente_ : consente agli utenti autorizzati
          di accedere al sistema
       - quotazione dei prezzi: consente che la vendita e l'acquisto
          degli stock dei beni siano quotati
       - piazzamento ordini: consente di vendere o comprare beni
          al dato prezzo di mercato

- Questi componenti fanno uso dei dati

- relativi agli utenti
- accedono al database delle transazioni


=== Esempio

- Possibili attacchi al sistema:
    - utente malevolo guadagna le credenziali di accesso
       di utente verso cui nutre forte rancore
       - vengono piazzati ordini di vendita e acquisto in modo
          tali da causare seri problemi all'utente legittimo
    - utente non autorizzato corrompe il database
       delle transazioni guadagnando permessi
       per eseguire direttamente query SQL
       - riconciliare gli acquisti e le vendite diventa quindi impossibile


=== Esempio

#cfigure("images/2024-04-24-17-01-24.png",100%)

=== Capacità di Sopravvivenza

- Aggiungere le tecniche di sopravvivenza costa soldi
- Spesso le aziende sono molto riluttanti ad investire
    sulla sopravvivenza, specie se non hanno mai subito
    attacchi e perdite
- È comunque sempre buona norma investire
    nella sopravvivenza prima piuttosto che dopo aver già
    subito un attacco
- L'analisi della sopravvivenza non è ancora inclusa
    nella maggior parte dei processi di ingegnerizzazione
    del software
- Con la crescita dei sistemi critici sembra probabile
    che questo tipo di analisi sarà sempre più utilizzato



== Conclusioni

- La sicurezza deve #text(blue)[*essere onnipresente
    attraverso tutto il ciclo di sviluppo del software*]
- Inoltre la sicurezza deve essere tenuta in considerazione
    anche attraverso tutti gli strati dell'infrastruttura
    su cui l'applicazione viene sviluppata
- Per ottenere questo è imperativa l'adozione
    di un processo che tenga in considerazione
    le problematiche relative alla sicurezza
    sin dalle prime fasi dello sviluppo del sistema
- È inoltre necessario che vengano compiuti #text(blue)[*severi test
    periodici*] a verifica del livello di sicurezza del sistema





