#import "@preview/orionotes:0.1.0": orionotes

#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *



//Codly
#show: codly-init.with()

// Disabilita la funzione smallcaps per le celle (di tabelle)
#show smallcaps: it => it.body

#codly(
  languages: (
    rust: (name: "Rust", icon: "🦀", color: rgb("#CE412B")),
    py: (name: "Python", icon: "🐍", color: rgb("#4CAF50")),
    sh: (name: "Shell", icon: "💲", color: rgb("#89E051")),
    cpp: (
      name: "",
      icon: box(
        fill: rgb("#22c55e"),
        inset: (x: 0.3em, y: 0.15em),
        radius: 0.2em,
        text(fill: white, weight: "bold", size: 0.8em)[CUDA]
      ),
      color: rgb("#22c55e")
    ),
  ),
  // number-format: num => text(fill: rgb("#a8a29e"), str(num)),
  number-format: none,
  zebra-fill: none,
  stroke: none,
  fill: rgb("#faf8f5"),
  radius: 0.5em,
  // number-placement: "outside"
)





#show: orionotes.with(
  title: [Sistemi di Elaborazione Accelerata],
  authors: ("Bumma Giuseppe",),
  professors: ("Mattoccia Stefano", "Tosi Fabio"),
  date: "2025/2026",
  university: [Università degli studi di Bologna],
  degree: [Ingegneria Informatica Magistrale],
  // If you want it insert and image object
  front-image: none,
  preface: [The preface to your notes],
  appendix: (
    enabled: true,
    title: "Appendices",
    body: [
      // = Example
      // Here go the appendices.
    ]
  )
)

#set heading(numbering: "1.1.1.1.1.1")

// Variables
#let dark_orange = rgb("#e6a700")
#let light_orange = rgb("#FF9900")
#let light_green = rgb("#85B911")
#let light_blue = rgb("#3D85C6")
#let nvidia-green = rgb("#76B900")
#let device-blue = rgb("#4A90E2")
#let my_gray = rgb("#F3F3F3")


// Functions
#let green_t(body) = {
  text(fill: light_green)[#body]
}

#let blue_t(body) = {
  text(fill: light_blue)[#body]
}

#let green_heading(body) = {
  text(weight: "bold", fill: light_green)[#body]
}

#let blue_heading(body) = {
  text(weight: "bold", fill: light_blue)[#body]
}

#let orange_heading(body) = {
  text(weight: "bold", fill: light_orange)[#body]
}



#set text(9pt)
#set text(font: "Segoe UI")
#show raw.where(block: false): set text(size: 8pt)

// Funzione helper per cerchiare il testo
#let circled(body) = box(
  stroke: (paint: red, dash: "dashed", thickness: 1pt),
  radius: 100%, // Rende il box ovale
  inset: (x: 3pt, y: 0pt), // Spazio laterale interno
  outset: (y: 2.5pt), // Estende il bordo verticalmente per creare l'ovale senza muovere il testo
  body
)

#let dashed-line = grid.cell(
  colspan: 3, 
  line(length: 100%, stroke: (dash: "dashed", thickness: 0.5pt, paint: gray))
)


= Introduzione

== Perché Scegliere la Piattaforma CUDA?

*Dominio di Mercato e Standard Industriale*

Standard de facto per calcolo parallelo e GPU, ampiamente supportata da software e librerie come TensorFlow e PyTorch. Le GPU NVIDIA sono prevalenti in HPC, AI e simulazione scientifica.

*Prestazioni Elevate*

CUDA consente di sfruttare la potenza delle GPU NVIDIA per eseguire milioni di thread simultaneamente, migliorando significativamente le prestazioni rispetto ai processori CPU tradizionali.

*Ampia Documentazione e Risorse*

NVIDIA fornisce una documentazione dettagliata e risorse pratiche, mentre la comunità attiva consente il supporto e la condivisione di conoscenze tra sviluppatori.

*Facilità d'Uso*

CUDA estende i linguaggi di programmazione C, C++, e Fortran, permettendo agli sviluppatori di utilizzare sintassi e concetti già noti.

*Versatilità*

È utilizzato in vari campi, dalla grafica 3D alla simulazione scientifica, dall'elaborazione video al deep learning, rendendolo una scelta flessibile per molti progetti.

*Ecosistema Ricco*

CUDA offre un ampio set di librerie ottimizzate (cuBLAS, cuDNN, Thrust, etc.) e strumenti di sviluppo per facilitare l'ottimizzazione e il debugging.

== Cos'è il CUDA Toolkit?

- Il CUDA Toolkit è un insieme completo di strumenti di sviluppo fornito da NVIDIA per creare applicazioni accelerate tramite GPU.
- È essenziale per lo sviluppo di applicazioni CUDA, poiché fornisce tutti gli strumenti necessari per scrivere, compilare e ottimizzare codice CUDA.

=== Componenti chiave del CUDA Toolkit

*Driver NVIDIA*

- Fondamento Invisibile: Essenziali per CUDA, ma gli sviluppatori raramente interagiscono direttamente con essi.
- Ruolo: Funzionano da ponte tra il sistema operativo e la GPU, gestendo l'hardware, il caricamento del codice e il trasferimento dati tra CPU e GPU.
- Installazione: Necessari per CUDA e di solito installati separatamente.
- Compatibilità: Devono essere compatibili sia con la versione del CUDA Toolkit utilizzata che con la GPU in uso.

*CUDA Runtime / CUDA Driver API (Application Programming Interface)*

- Gli sviluppatori possono scegliere tra due interfacce per interagire con la GPU:
  - CUDA Runtime API: Livello di astrazione più alto, più semplice da utilizzare.
  - CUDA Driver API: Livello di astrazione più basso, offre un controllo più granulare sulle operazioni.

*Compilatore CUDA (NVIDIA CUDA Compiler - `nvcc`)*

- Traduzione del codice: Compila il codice CUDA, scritto in linguaggi come C, C++ e Fortran, in un formato eseguibile dalle GPU NVIDIA.
- Fasi:
  - Separazione: Separa il codice destinato alla CPU da quello per la GPU.
  - Compilazione: Compila il codice GPU in PTX (linguaggio intermedio) o direttamente in codice macchina per l'architettura GPU target (tenendo conto della Compute Capability e della CUDA Version utilizzata).
  - Linking: Combina il codice CPU e GPU con le librerie CUDA per creare l'applicazione finale.

*Librerie CUDA*

- Le librerie forniscono implementazioni ottimizzate e parallele di operazioni comuni, così gli sviluppatori non devono reinventare algoritmi complessi da zero.
- Esempi:
  - cuBLAS: Algebra lineare (operazioni su matrici e vettori).
  - cuFFT: Fast Fourier Transform (analisi di segnali, elaborazione di immagini).
  - cuDNN: Primitive per deep neural networks (convoluzione, pooling, attivazione).
  - cuRAND: Generazione di numeri casuali (simulazioni Monte Carlo, crittografia).
  - cuSPARSE: Operazioni su matrici sparse (risoluzione di sistemi lineari sparsi).
  - Thrust: Algoritmi paralleli generici (ordinamento, ricerca, trasformazioni) su GPU.

*Esempi di Codice (CUDA Samples)*

- Utilità: Forniscono implementazioni concrete di algoritmi e applicazioni comuni che utilizzano CUDA.
- Scopo: Aiutano gli sviluppatori a imparare le best practice di programmazione CUDA e a iniziare rapidamente nuovi progetti.

==== Strumenti di Debugging e Profiling

- Nsight Systems:
  - Profiling a livello di sistema. Offre una visione d'insieme del comportamento dell'applicazione su CPU e GPU, evidenziando eventuali colli di bottiglia.
- Nsight Compute:
  - Profiling approfondito della GPU. Permette di analizzare le prestazioni dei kernel CUDA in dettaglio, identificando aree di ottimizzazione per la memoria e l'utilizzo dei core.
- CUDA-GDB:
  - Debugging a riga di comando. Permette di eseguire il debug del codice CUDA a livello di sorgente, sia sulla CPU che sulla GPU. Supporta breakpoint, ispezione di variabili e stack trace su thread GPU.
- NVIDIA Visual Profiler (NVVP) [non più supportato]:
  - Offre una rappresentazione grafica timeline delle attività della CPU e della GPU, facilitando l'identificazione dei colli di bottiglia. Oggi le sue funzionalità sono state integrate in Nsight.

==== Relazione tra CUDA Toolkit e CUDA Version

- Il CUDA Toolkit viene rilasciato in versioni numerate (es. CUDA 12.6, CUDA 11.0). La versione installata determina la CUDA Version in uso sul sistema.
- Ogni nuova CUDA Version include aggiornamenti software e bugfix, nuovi strumenti di sviluppo e librerie, supporto per le più recenti architetture GPU NVIDIA.
- Aggiornare il CUDA Toolkit alla versione più recente garantisce compatibilità con le GPU più recenti e l'accesso alle ultime ottimizzazioni, migliorando performance e stabilità del codice.

==== Retrocompatibilità del CUDA Toolkit

- Supporto per GPU Precedenti: Le nuove versioni del CUDA Toolkit mantengono la compatibilità con GPU più vecchie, anche se non tutte le funzionalità più recenti sono disponibili su queste architetture.
- Limitazioni: Alcune funzionalità avanzate introdotte nelle nuove versioni potrebbero non essere supportate su GPU datate, e il supporto per architetture molto vecchie può essere gradualmente ridotto o deprecato.
- Compatibilità del Codice: In generale, il codice scritto con versioni precedenti del Toolkit può essere eseguito su versioni più recenti, ma può richiedere piccoli adattamenti.

== CUDA Compute Capability (CC)

- La Compute Capability (CC) è un numero in formato X.Y che identifica le caratteristiche e le capacità di una GPU NVIDIA in termini di funzionalità supportate e limiti hardware
- Il numero X (principale) indica la generazione dell'architettura (ad es. Turing, Ampere), mentre Y (secondario) indica una revisione della stessa architettura, con piccoli miglioramenti o varianti hardware.

#figure(image("images/_page_62_Figure_3.jpeg"))



=== Relazione tra Compute Capability (CC) e CUDA Version

- *Compute Capability (CC)*
  - Indica le caratteristiche e i limiti hardware di una GPU.
  - È indipendente dalla CUDA Version, ma la CUDA Version deve supportare la
      Compute Capability della GPU per sfruttare appieno le sue capacità.
- *CUDA Version*
  - Determina quali funzionalità software, API e librerie sono disponibili per lo sviluppo.
  - Può supportare più Compute Capabilities: una singola versione di CUDA Toolkit può essere compatibile con diverse generazioni di GPU (es. CUDA 11.x supporta Volta, Turing, Ampere).




== Evoluzione delle Architetture GPU NVIDIA

- Progressione Tecnologica: Da Fermi a Blackwell, ogni generazione ha portato significativi avanzamenti nelle capacità di calcolo e nell'efficienza energetica.
- Adattamento al Mercato: L'evoluzione riflette il passaggio da un focus su grafica e HPC a un'enfasi crescente su AI, deep learning e calcolo ad alte prestazioni.

#figure(image("images/_page_64_Figure_3.jpeg"))

== Anatomia di un Programma CUDA

=== Struttura del Codice Sorgente

- File Sorgente: Estensione `.cu`
- Codice host + Codice device


=== Componenti Principali

- Codice Host
  - Codice C/C++ eseguito sulla CPU
  - Gestisce la logica dell'applicazione
  - Alloca memoria sulla GPU
  - Trasferisce dati tra CPU e GPU
  - Lancia i kernel GPU
  - Gestisce la sincronizzazione
- Codice Device:
  - Codice CUDA C eseguito sulla GPU
  - Contiene i kernel (funzioni parallele)
  - Esegue operazioni computazionali intensive in parallelo

#figure(image("images/_page_65_Picture_16.jpeg"))


=== Flusso di Compilazione

- Separazione del codice
  - Il compilatore NVIDIA CUDA Compiler (`nvcc`) separa il codice device dal codice host
- Compilazione del codice host
  - È codice C standard o C++
  - Compilato con compilatori C tradizionali (gcc)
- Compilazione del codice device
  - Compilato da `nvcc` in formato intermedio PTX (Parallel Thread Execution) .
  - Il driver NVIDIA poi traduce il PTX in codice macchina specifico per la GPU (SASS - Streaming Assembly) al momento dell'esecuzione, usando un compilatore Just-In-Time (JIT).
- Linking
  - Aggiunta delle librerie runtime CUDA
  - Supporto per chiamate ai kernel e manipolazione esplicita della GPU
- Eseguibile finale
  - File unico con codice per CPU e GPU


== Compilazione di un Programma CUDA

#figure(image("images/_page_67_Figure_1.jpeg"))

== Hello World in CUDA C

=== Passo 1: Creare il File Sorgente

Nome file: `hello.cu`

==== Passo 2: Scrivere il codice

Codice C e CUDA C a confronto


#figure(
  // caption: "Linguaggio C"
)[
  #codly(header: [#align(center)[*Linguaggio C*]])
  ```c
  include <stdio.h>
    int main(void) {
      printf("Hello World from CPU!\n");
      return 0;
    }
  ```
] <Linguaggio-C>

#figure(
)[
  #codly(header: [#align(center)[*Linguaggio CUDA C*]])
  ```cpp
  include <stdio.h>
  __global__ void helloFromGPU()
  {
    printf("Hello World from GPU thread 
    %d!\n", threadIdx.x);
  }
  int main()
  {
    // Lancio del kernel
    helloFromGPU<<<1, 10>>>();
    // Attendere che la GPU finisca
    cudaDeviceSynchronize();
    return 0;
  }
  ```
] <Linguaggio-CUDA-C>


Linguaggio CUDA C

== Hello World in CUDA C - Analisi

- Definizione kernel GPU:
  - `__global__`: Qualificatore CUDA che indica una funzione eseguita sulla GPU, ma chiamata dalla CPU. (In C standard non esiste).
  - threadIdx.x: Variabile built-in CUDA che fornisce l'ID univoco del thread all'interno del blocco.

```cpp
__global__ void helloFromGPU(){
 printf("Hello World from GPU thread %d!\n", threadIdx.x);}
```

- Funzione `main`: Punto di ingresso del programma, eseguito sulla CPU come in C standard.
- Lancio del kernel
  - `<<<1, 10>>>`: Configurazione di esecuzione (1 blocco, 10 thread). Avvia 10 istanze parallele del kernel sulla GPU.
- Sincronizzazione GPU-CPU
  - `cudaDeviceSynchronize()`: la CPU attende che la GPU completi tutte le operazioni prima di proseguire/terminare.

```cpp
int main(){
  helloFromGPU<<<1, 10>>>();
  cudaDeviceSynchronize();
  return 0;
}
```

== Hello World in CUDA C

==== Passo 3: Compilazione

- Salvare il codice nel file `hello.cu`
  - Un file `.cu` può contenere sia codice C/C++ standard che codice CUDA C
  - Questo permette di mescolare codice per CPU e GPU nello stesso file
- Compilare il programma usando il compilatore CUDA `nvcc`
  - `nvcc` può gestire sia il codice C/C++ standard che le estensioni CUDA
  - `nvcc` separa internamente il codice host e device, compilando ciascuno in modo appropriato

```sh
$ nvcc hello.cu -o hello
```

== Hello World in CUDA C

==== Passo 4: Esecuzione

Eseguire il file eseguibile:

```
$ ./hello
```

Output:

#figure(
)[
  #codly(header: [#align(center)[*Variante Linguaggio C*]])
  ```
  Hello World from CPU!
  ```
]


#figure(
)[
  #codly(header: [#align(center)[*Variante Linguaggio CUDA C*]])
  ```
  Hello World from GPU thread 0! 
  Hello World from GPU thread 1!
  Hello World from GPU thread 2!
  Hello World from GPU thread 3!
  Hello World from GPU thread 4!
  Hello World from GPU thread 5!
  Hello World from GPU thread 6!
  Hello World from GPU thread 7!
  Hello World from GPU thread 8!
  Hello World from GPU thread 9!
  ```
]




== Ottenere Informazioni sulla GPU tramite API CUDA

=== Utilizzo delle API CUDA

CUDA fornisce API per ottenere informazioni dettagliate sulle GPU direttamente dal codice

```cpp
int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0); // Ottieni proprietà del dispositivo 0
  printf("Nome Dispositivo: %s\n", prop.name);
  printf("Memoria Globale Totale: %.0f MB\n", prop.totalGlobalMem / 1024.0 / 1024.0);
  printf("Clock Core: %d MHz\n", prop.clockRate / 1000);
  printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
  // Stampa di altre proprietà
  return 0;
}
```

- Utilizzo della struttura `cudaDeviceProp` per memorizzare le proprietà del device
- Utilizzo della funzione `cudaGetDeviceProperties` per ottenere le proprietà del device specificato
- Accesso alle proprietà della GPU, come nome, memoria totale, clock core e compute capability
- Per una lista completa delle proprietà disponibili, consulta la documentazione ufficiale di CUDA(https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html)

== Ottenere Informazioni sulla GPU tramite API CUDA

==== Esempio di Output

```
CUDA System Information:
CUDA Driver Version: 12.2
CUDA Runtime Version: 11.3
Numero di dispositivi CUDA: 2
Dispositivo 0: NVIDIA GeForce RTX 3090
1. Compute Capability: 8.6
2. Memoria Globale Totale: 23.69 GB
3. Numero di Multiprocessori: 82
4. Clock Core: 1695 MHz
5. Clock Memoria: 9751 MHz
6. Larghezza Bus Memoria: 384 bit
7. Dimensione Cache L2: 6144 KB
8. Memoria Condivisa per Blocco: 48 KB
9. Numero Massimo di Thread per Blocco: 1024
10. Dimensioni Massime Griglia: (2147483647, 65535, 65535)
11. Dimensioni Massime Blocco: (1024, 1024, 64)
12. Warp Size: 32
13. Memoria Costante Totale: 65536 bytes
14. Texture Alignment: 512 bytes
```

== Cosa Significa Programmare in CUDA?

=== Pensare in Parallelo

- *Decomposizione del Problema:* Identificare le parti del problema che possono essere eseguite in parallelo per sfruttare al meglio le risorse della GPU.
- *Architettura della GPU:* Le GPU sono composte da migliaia di core in grado di eseguire thread in parallelo. CUDA fornisce gli strumenti per organizzare e gestire questi thread.
- *Scalabilità:* Progettare algoritmi che si adattano a diversi numeri di thread (e GPU).
- *Gerarchia di Thread:* Organizzare il lavoro in blocchi e griglie per massimizzare l'efficienza.
- *Gerarchia di Memoria:* Utilizzare strategicamente memoria globale, condivisa, locale e registri per ridurre i tempi di accesso.
- *Sincronizzazione:* Gestire la coordinazione tra thread e il trasferimento dati tra CPU e GPU senza conflitti.
- *Bilanciamento del Carico:* Distribuire il lavoro in modo uniforme fra thread per evitare colli di bottiglia.

=== Scrittura di codice in CUDA C

- CUDA estende C/C++ con costrutti specifici come `__global__`, `__shared__` e la sintassi `<<<...>>>` per lanciare kernel sulla GPU.
- Ogni kernel viene scritto come codice sequenziale, ma viene eseguito in parallelo da migliaia di thread, permettendo di pensare in modo semplice ma scalare.


==== Testi Generali

- Cheng, J., Grossman, M., McKercher, T. (2014). Professional CUDA C Programming. Wrox Pr Inc. (1^ edizione)
- Kirk, D. B., Hwu, W. W. (2022). Programming Massively Parallel Processors. Morgan Kaufmann (4^ edizione)

==== NVIDIA Docs

- CUDA Programming:
  - http://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA C Best Practices Guide
  - http://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- CUDA University Courses
  - https://developer.nvidia.com/educators/existing-courses=2
- An Even Easier Introduction to CUDA
  - https://developer.nvidia.com/blog/even-easier-introduction-cuda/

==== Materiale di Approfondimento

- Branch Education (canale YouTube con spiegazioni su GPU, ray tracing, hardware, ecc.)
  - https://www.youtube.com/@BranchEducation



= Modello di Programmazione CUDA



== La Struttura Stratificata dell'Ecosistema CUDA

#figure(image("images/_page_3_Figure_1.png", width: 70%))

Ecosistema stratificato per algoritmi paralleli su GPU, con semplicità e controllo hardware ottimizzati.


Applicazioni: Programmi scritti dagli sviluppatori per risolvere problemi specifici utilizzando CUDA.

Librerie: Raccolte di funzioni ottimizzate (es. cuBLAS, cuDNN) che semplificano lo sviluppo.

Modello di Programmazione: CUDA fornisce un'astrazione per la programmazione GPU, offrendo concetti come thread, blocchi e griglie.

Compilatore: Strumenti (`nvcc`) che traducono il codice in istruzioni GPU eseguibili.

Driver CUDA + Sistema Operativo: Il sistema operativo gestisce le risorse; il driver CUDA traduce le chiamate CUDA in comandi per la GPU.

Architetture: Le specifiche GPU NVIDIA su cui il codice CUDA viene eseguito, con diverse capacità e caratteristiche.

== Ruolo del Modello e del Programma

*Il Modello di Programmazione:*

Definisce la struttura e le regole per sviluppare applicazioni parallele su GPU. Elementi fondamentali:

- Gerarchia di Thread: Organizza l'esecuzione parallela in thread, blocchi e griglie, ottimizzando la scalabilità su diverse GPU.
- Gerarchia di Memoria: Offre tipi di memoria (globale, condivisa, locale, costante, texture) con diverse prestazioni e scopi, per ottimizzare l'accesso ai dati.
- API: Fornisce funzioni e librerie per gestire l'esecuzione del kernel, il trasferimento dei dati e altre operazioni essenziali.

*Il Programma:*

Rappresenta l'implementazione concreta (il codice) che specifica come i thread condividono dati e coordinano le loro attività. Nel programma CUDA, si definisce:

- Come i dati verranno suddivisi e elaborati tra i vari thread.
- Come i thread accederanno alla memoria e condivideranno dati.
- Quali operazioni verranno eseguite in parallelo.
- Quando e come i thread si sincronizzeranno per completare un compito.

== Livelli di Astrazione nella Programmazione Parallela CUDA

Il calcolo parallelo si articola in tre livelli di astrazione: dominio, logico e hardware, guidando l'approccio del programmatore.


=== Livello Dominio

- Focus sulla decomposizione del problema.
- Definizione della struttura parallela di alto livello.

Chiave: Ottimizza la strategia di parallelizzazione.


=== Livello Logico

- Organizzazione e gestione dei thread.
- Implementazione della strategia di parallelizzazione.

Chiave: Massimizza l'efficienza del parallelismo.

==== Livello Hardware

- Mappatura dell'esecuzione sull'architettura GPU.
- Ottimizzazione delle prestazioni hardware.

Chiave: Sfrutta al meglio le risorse GPU.

#figure(image("images/_page_4_Picture_16.png", width: 80%))




== Thread CUDA: L'Unità Fondamentale di Calcolo

=== Cos'è un Thread CUDA?

- Un thread CUDA rappresenta un'unità di esecuzione elementare nella GPU.
- Ogni thread CUDA esegue una porzione di un programma parallelo, chiamato kernel.
- Sebbene migliaia di thread vengano eseguiti concorrentemente sulla GPU, ogni singolo thread segue un percorso di esecuzione sequenziale all'interno del suo contesto.


=== Cosa Fa un Thread CUDA?

- *Elaborazione di Dati:* Ogni thread CUDA si occupa di un piccolo pezzo del problema complessivo, eseguendo calcoli su un sottoinsieme di dati.
- *Esecuzione di Kernel:* Ogni thread esegue lo stesso codice del kernel ma opera su dati diversi, determinati dai suoi identificatori univoci (threadIdx,blockIdx).
- *Stato del Thread:* Ogni thread ha il proprio stato, che include il program counter, i registri, la memoria locale e altre risorse specifiche del thread.

*Thread CUDA vs Thread CPU*

- *GPU:* parallelismo massivo (migliaia di core leggeri), basso overhead di gestione.
- *CPU:* parallelismo limitato (pochi core complessi), overhead più elevato.

== Struttura di Programmazione CUDA

#figure(image("images/_page_6_Figure_1.jpeg"))

=== Caratteristiche Principali

- *Codice Seriale e Parallelo:* Alternanza tra sezioni di codice seriale e parallelo (stesso file).
- *Struttura Ibrida Host-Device:* Alternanza tra codice eseguito sulla CPU (host) e sulla GPU (device).
- *Esecuzione Asincrona:* Il codice host può continuare l'esecuzione mentre i kernel GPU sono in esecuzione.
- *Kernel CUDA Multipli:* Possibilità di lanciare più kernel nella stessa applicazione, anche in overlapping temporale.
- *Gestione dei Risultati sull'Host:* Fase dedicata all'elaborazione dei risultati sulla CPU dopo l'esecuzione dei kernel.



== Flusso Tipico di Elaborazione CUDA

1. Inizializzazione e Allocazione Memoria (Host)

  Preparazione dati e allocazione di memoria su CPU (host) e GPU (device).

2. Trasferimento Dati (Host → Device)

  Copia degli input dalla memoria host alla memoria device.

3. Esecuzione del Kernel (Device)

  La GPU esegue calcoli paralleli secondo la configurazione di griglia e blocchi.

4. Recupero Risultati (Device → Host)

  Copia dell'output dalla memoria device alla memoria host.

5. Post-elaborazione (Host)

  Analisi o elaborazione aggiuntiva dei risultati sulla CPU.

6. Liberazione Risorse

  Rilascio della memoria allocata su host e device.

*Nota:* i passi 2-5 possono essere ripetuti più volte o eseguiti in pipeline tramite stream per massimizzare l'overlap tra calcolo e trasferimento dati.



== Gestione della Memoria in CUDA

=== Modello di Memoria CUDA

- Il modello prevede un sistema con host (CPU) e device (GPU), ciascuno con la propria memoria.
- La comunicazione tra memoria host e device avviene tramite PCIe (Peripheral Component Interconnect Express), interfaccia seriale point-to-point che sfrutta più lane indipendenti in parallelo per aumentare la banda.

=== Caratteristiche PCIe

- *Lane:* Ogni lane (canale di trasmissione) è costituito da due coppie di segnali differenziali (quattro fili), una per ricevere (RX) e una per trasmettere (TX) dati.
- *Full-Duplex:* Trasmette e riceve dati simultaneamente in entrambe le direzioni.
- *Scalabilità:* La larghezza di banda varia a seconda del numero di lane (x1, x2, x4, x8, x16).
- *Bassa Latenza:* Garantisce trasferimenti rapidi, adatti a scambi frequenti.
- *Collo di Bottiglia:* Può diventare un collo di bottiglia in trasferimenti di grandi volumi tra CPU e GPU.

#figure(image("images/_page_9_Picture_10.jpeg", width: 50%))




== Collegamento Fisico della GPU tramite PCIe

=== Connessione Fisica GPU

- La GPU si collega alla scheda madre attraverso uno slot PCI Express (PCIe).
- Il connettore, costituito da contatti metallici dorati sul bordo della scheda, si inserisce nello slot PCIe corrispondente.
- La maggior parte delle schede madri moderne ha uno o più slot PCIe, generalmente con almeno uno slot PCIe x16 destinato alla GPU.

#figure(image("images/_page_11_Picture_5.jpeg"))


== Modello di Memoria CUDA

- I kernel CUDA operano sulla memoria del device.
- CUDA Runtime fornisce funzioni per:
  - Allocare memoria sul device.
  - Rilasciare memoria sul device quando non più necessaria.
  - Trasferire dati bidirezionalmente tra la memoria dell'host e quella del device.

#figure(image("images/_page13_.png"))

*Nota Importante:* è responsabilità del programmatore gestire correttamente l'allocazione, il trasferimento e la deallocazione della memoria per ottimizzare le prestazioni.

== Gerarchia di Memoria

In CUDA, esistono diversi tipi di memoria, ciascuno con caratteristiche specifiche in termini di accesso, velocità, e visibilità. Per ora, ci concentriamo su due delle più importanti:

*Global Memory*

- Accessibile da tutti i thread su tutti i blocchi
- Più grande ma più lenta rispetto alla shared memory
- Persiste per tutta la durata del programma CUDA
- È adatta per memorizzare dati grandi e persistenti

*Shared Memory*

- Condivisa tra i thread all'interno di un singolo blocco
- Più veloce, ma limitata in dimensioni
- Esiste solo per la durata del blocco di thread
- Utilizzata per dati temporanei e intermedi

*Funzioni*

- `cudaMalloc`: Alloca memoria sulla GPU.
- `cudaMemcpy`: Trasferisce dati tra host e device.
- `cudaMemset`: Inizializza la memoria del device.
- `cudaFree`: Libera la memoria allocata sul device.

Nota: Queste funzioni operano principalmente sulla Global Memory.

#figure(image("images/_page_13_Figure_19.jpeg"))

== Allocazione della Memoria sul Device


`cudaMalloc` è una funzione CUDA utilizzata per allocare memoria sulla GPU (device).

Firma della Funzione (#link("https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html=group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356")[Documentazione Online])
```cpp
cudaError_t cudaMalloc(void devPtr, size_t size)
```

*Parametri*

- `devPtr`: Puntatore doppio che conterrà l'indirizzo della memoria allocata sulla GPU.
- `size`: Dimensione in byte della memoria da allocare.

*Valore di Ritorno*

`cudaError_t`: codice di errore (`cudaSuccess` se l'allocazione ha successo).

*Note Importanti*

- *Allocazione:* Riserva memoria lineare contigua sulla GPU a runtime.
- *Puntatore:* Aggiorna puntatore CPU con indirizzo memoria GPU.
- *Stato iniziale:* La memoria allocata non è inizializzata.



`cudaMemset` è una funzione CUDA utilizzata per impostare un valore specifico in un blocco di memoria allocato sulla GPU (device).

*Firma della Funzione* (#link("https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html=group__CUDART__MEMORY_1gf7338650f7683c51ee26aadc6973c63a")[Documentazione Online])

```cpp
cudaError_t cudaMemset(void devPtr, int value, size_t count)
```

*Parametri*

- `devPtr`: Puntatore alla memoria allocata sulla GPU.
- `value`: Valore da impostare in ogni byte della memoria.
- `count`: Numero di byte della memoria da impostare al valore specificato.

*Valore di Ritorno*

`cudaError_t`: Codice di errore (cudaSuccess se l'inizializzazione ha successo).

*Note Importanti*

- *Utilizzo:* Comunemente utilizzata per azzerare la memoria (impostando `value` a 0).
- *Gestione:* L'inizializzazione deve avvenire dopo l'allocazione della memoria tramite cudaMalloc.
- *Efficienza:* È preferibile usare `cudaMemset` per grandi blocchi di memoria per ridurre l'overhead.



=== Esempio di Allocazione di Memoria sulla GPU

Mostra come allocare memoria sulla GPU utilizzando `cudaMalloc`.

```cpp
float d_array; // Dichiarazione di un puntatore per la memoria sul device (GPU)
size_t size = 10  sizeof(float); // Calcola la dimensione della memoria da allocare (10 float)
// Allocazione della memoria sul device 
cudaError_t err = cudaMalloc((void)&d_array, size);
// Controlla se l'allocazione della memoria ha avuto successo
if (err != cudaSuccess) {
  // Se c'è un errore, stampa un messaggio di errore con la descrizione dell'errore
  printf("Errore nell'allocazione della memoria: %s\n", cudaGetErrorString(err));
} else {
  // Se l'allocazione ha successo, stampa un messaggio di conferma
  printf("Memoria allocata con successo sulla GPU.\n");}
```

== Trasferimento Dati


cudaMemcpy è una funzione CUDA per il trasferimento di dati tra la memoria dell'host e del device, o all'interno dello stesso tipo di memoria.

*Firma della Funzione* (#link("https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html=group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8")[Documentazione Online])

```cpp
cudaError_t cudaMemcpy(void dst, const void src, size_t count, cudaMemcpyKind kind)
```

*Parametri*

- `dst`: Puntatore alla memoria di destinazione.
- `src`: Puntatore alla memoria sorgente.
- `count`: Numero di byte da copiare.
- `kind`: Direzione della copia (cudaMemcpyKind).

*Tipi di Trasferimento (kind)*

- `cudaMemcpyHostToHost`: Da host a host
- `cudaMemcpyHostToDevice`: Da host a device
- `cudaMemcpyDeviceToHost`: Da device a host
- `cudaMemcpyDeviceToDevice`: Da device a device

*Valore di Ritorno*

`cudaError_t`: Codice di errore (`cudaSuccess` se il trasferimento ha successo).

*Note importanti*

- Funzione sincrona: blocca l'host fino al completamento del trasferimento.
- Per prestazioni ottimali, minimizzare i trasferimenti tra host e device.


=== Spazi di Memoria Differenti

Attenzione: I puntatori del device non devono essere dereferenziati nel codice host (spazi di memoria CPU e GPU differenti).

Esempio: assegnazione errata come
```cpp
host_array = dev_ptr
```

invece di

```cpp
cudaMemcpy(host_array, dev_ptr, nBytes, cudaMemcpyDeviceToHost)
```

*Conseguenza dell'errore:* Accesso a indirizzi non validi → possibile blocco o crash dell'applicazione.

*Soluzione:* La Unified Memory, introdotta in CUDA 6 e oggi ottimizzata, consente di usare un unico puntatore valido per CPU e GPU, con gestione automatica della migrazione dati (vedremo in seguito).



== Deallocazione della Memoria sul Device

`cudaFree` è una funzione CUDA utilizzata per liberare la memoria precedentemente allocata sulla GPU (device).

*Firma della Funzione* (#link("https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html=group__CUDART__MEMORY_1ga042655cbbf3408f01061652a075e094")[Documentazione Online]) 

```cpp
cudaError_t cudaFree(void devPtr)
```

*Parametri*

`devPtr`: Puntatore alla memoria sul device che deve essere liberata. Questo puntatore deve essere stato precedentemente restituito tramite la chiamata cudaMalloc.

*Valore di Ritorno*

`cudaError_t`: Codice di errore (`cudaSuccess` se la deallocazione ha successo).

*Note Importanti*

- *Gestione:* È responsabilità del programmatore assicurarsi che ogni blocco di memoria allocato con cudaMalloc sia liberato per evitare perdite di memoria (memory leaks) sulla GPU.
- *Efficienza:* La deallocazione è sincrona e può avere overhead significativo; è consigliato minimizzare il numero di chiamate.



=== Esempio di Allocazione e Trasferimento Dati

Mostra come allocare e trasferire dati dalla memoria host alla memoria device.

```cpp
size_t size = 10  sizeof(float); // Calcola la dimensione della memoria da allocare (10 float)
float h_data = (float)malloc(size); // Alloca memoria sull'host (CPU) per memorizzare i dati
for (int i = 0; i < 10; ++i) h_data[i] = (float)i; // Inizializza ogni elemento di h_data
float d_data; // Dichiarazione di un puntatore per la memoria sulla GPU (device)
cudaMalloc((void)&d_data, size); // Allocazione della memoria sulla GPU
// Copia dei dati dalla memoria dell'host (CPU) alla memoria del device (GPU)
cudaError_t err = cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
// Controlla se la copia è avvenuta con successo
if (err != cudaSuccess) {
  // Se c'è un errore, stampa un messaggio di errore e termina il programma
  fprintf(stderr, "Errore nella copia H2D: %s\n", cudaGetErrorString(err));
  exit(EXIT_FAILURE);
}

// Esegui operazioni sulla memoria della GPU (d_data)
// (Le operazioni specifiche da eseguire non sono mostrate in questo esempio)
// Copia dei risultati dalla memoria della GPU (device) alla memoria dell'host (CPU)
err = cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
// Controlla se la copia è avvenuta con successo
if (err != cudaSuccess) {
  fprintf(stderr, "Errore nella copia D2H: %s\n", cudaGetErrorString(err));
  exit(EXIT_FAILURE);
}
free(h_data); // Libera la memoria allocata sull'host
cudaFree(d_data); // Libera la memoria allocata sulla GPU
```


== Organizzazione dei Thread in CUDA

CUDA adotta una #underline[*gerarchia a due livelli*] per organizzare i thread basata su blocchi di thread e griglie di blocchi.

=== Struttura Gerarchica

+ *Grid (Griglia)*

  - Array di thread blocks.
  - È organizzata in una struttura 1D, 2D o 3D.
  - Rappresenta l'intera computazione di un kernel.
  - Contiene tutti i thread che eseguono il singolo kernel.
  - Condivide lo stesso spazio di memoria globale.

+ *Block (Blocco)*

  - Un thread block è un gruppo di thread eseguiti logicamente in parallelo.
  - Ha un ID univoco all'interno della sua griglia.
  - I blocchi sono organizzati in una struttura 1D, 2D o 3D.
  - I thread di un blocco possono sincronizzarsi (non automaticamente) e condividere memoria.
  - I thread di blocchi diversi non possono sincronizzarsi  direttamente (solo tramite memoria globale o kernel successivi)

+ *Thread*

- Ha un proprio ID univoco all'interno del suo blocco.
- Ha accesso alla propria memoria privata (registri).

#figure(image("images/_page_23_Figure_18.jpeg", width: 70%))

=== Perché una Gerarchia di Thread?

*Mappatura Intuitiva*

La gerarchia di thread (grid, blocchi, thread) permette di scomporre problemi complessi in unità di lavoro parallele più piccole e gestibili, rispecchiando spesso la struttura intrinseca del problema stesso.

*Organizzazione e Ottimizzazione*

Il programmatore può definire le dimensioni dei blocchi e della griglia per adattare l'esecuzione alle caratteristiche specifiche dell'hardware e del problema, ottimizzando l'utilizzo delle risorse.

*Efficienza nella Memoria*

I thread in un blocco possono condividere dati tramite memoria on-chip veloce (es. shared memory), riducendo gli accessi alla memoria globale più lenta, migliorando dunque significativamente le prestazioni.

*Scalabilità e Portabilità*

La gerarchia è scalabile e permette di adattare l'esecuzione a GPU con diverse capacità e numero di core. Il codice CUDA, quindi, risulta più portabile e può essere eseguito su diverse architetture GPU.

*Sincronizzazione Granulare*

I thread possono essere sincronizzati solo all'interno del proprio blocco, evitando costose sincronizzazioni globali che possono creare colli di bottiglia.

== Identificazione dei Thread in CUDA

Ogni thread ha un'identità unica definita da coordinate specifiche nella gerarchia grid-block. Tali coordinate, diverse per ogni thread, sono essenziali per calcolare indici di lavoro e accedere correttamente ai dati.

#figure(image("images/_page26_2.1.png"))

*
Variabili di Identificazione (Coordinate)*

+ `blockIdx` (indice del blocco all'interno della griglia)
  - Componenti: `blockIdx.x,blockIdx.y,blockIdx.z`
+ `threadIdx` (indice del thread all'interno del blocco)
  - Componenti: `threadIdx.x,threadIdx.y,threadIdx.z`

Entrambe sono variabili built-in di tipo uint3 pre-inizializzate dal CUDA Runtime e accessibili solo all'interno del kernel.

*Variabili di Dimensioni*

+ `blockDim` (dimensione del blocco in termini di thread)
  - Tipo: `dim3` (lato host), `uint3` (lato device, built-in)
  - Componenti: `blockDim.x,blockDim.y,blockDim.z`
+ `gridDim` (dimensione della griglia in termini di blocchi)
  - Tipo: `dim3` (lato host), `uint3` (lato device, built-in)
  - Componenti: `gridDim.x,gridDim.y,gridDim.z`


`uint3` è un built-in vector type di CUDA con tre campi (x,y,z) ognuno di tipo unsigned int


=== Dimensione delle Griglie e dei Blocchi

- La scelta delle dimensioni ottimali dipende dalla struttura dati del problema e dalle capacità hardware/risorse della GPU.
- Le dimensioni di griglia e blocchi vengono definite nel codice host prima del lancio del kernel.
- Sia le griglie che i blocchi utilizzano il tipo dim3 (lato host) con tre campi unsigned int. I campi non utilizzati vengono inizializzati a 1 e ignorati.
- 9 possibili configurazioni (1D, 2D, 3D per griglia e blocco) in tutto anche se in genere si usa la stessa per entrambi.



== Struttura `dim3`

*Definizione*

- `dim3` è una struttura definita in `vector_types.h` usata per specificare le dimensioni di griglia e blocchi.
- Supporta le dimensioni 1, 2 e 3:

#figure(
)[
  #codly(header: [#align(center)[*Esempi*]])
  ```cpp
  dim3 gridDim(256); // Definisce una griglia di 256x1x1 blocchi.
  dim3 blockDim(512, 512);` // Definisce un blocco di 512x512x1 threads.
  ```
]


Utilizzato per specificare le dimensioni di griglia e blocchi quando si lancia un kernel dal lato host:

```cpp
kernel_name<<<gridDim, blockDim>>>(...);
```

*Codice Originale:*

```cpp
struct __device_builtin__ dim3
{
  unsigned int x, y, z;
  =if defined(__cplusplus)
  __host__ __device__ dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
  __host__ __device__ dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
  __host__ __device__ operator uint3(void) { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
  =endif / __cplusplus /
};
```

`gridDim.x`: Numero di blocchi nella griglia, in questo caso 3.



#figure(image("images/_page_28_Figure_3.jpeg"))

`blockDim.x`: Numero di thread per blocco, in questo caso 4.

#figure(image("images/_page_29_Figure_2.jpeg"))

`blockIdx.x`: Indice di un blocco nella griglia.

#figure(image("images/_page_30_Figure_2.jpeg"))

`threadIdx.x`: Indice del thread all'interno del blocco.

#figure(image("images/_page_31_Figure_2.jpeg"))

`gridDim.x`, `gridDim.y`: Numero di blocchi nella griglia lungo le dimensioni x e y.

#figure(image("images/_page_32_Figure_2.jpeg"))

`blockDim.x`, `blockDim.y`: Numero di thread per blocco lungo le dimensioni x e y.

#figure(image("images/_page_33_Figure_2.jpeg"))

`blockIdx.x`, `blockIdx.y`: Indici del blocco lungo le dimensioni x e y della griglia.

#figure(image("images/_page_34_Figure_2.jpeg"))

`threadIdx.x`, `threadIdx.y`: Indici x e y del thread nel blocco 2D.

#figure(image("images/_page_35_Figure_2.jpeg"))


== Esecuzione di un Kernel CUDA

=== Cos'è un Kernel CUDA?

- Un kernel CUDA è una funzione che viene eseguita in parallelo sulla GPU da migliaia o milioni di thread.
- Rappresenta il nucleo computazionale di un programma CUDA.
- Nei kernel viene definita la logica di calcolo per un singolo thread e l'accesso ai dati associati a quel thread.
- Ogni thread esegue lo stesso codice kernel, ma opera su diversi elementi dei dati.

*Sintassi della chiamata Kernel CUDA*

```cpp
kernel_name <<<gridSize,blockSize>>>(argument list); 
```

- `gridSize`:Dimensione della griglia (num. di blocchi).
- `blockSize`: Dimensione del blocco (num. di thread per blocco).
- `argument list`: Argomenti passati al kernel.

`Sintassi Standard C`

```c
function_name (argument list);
```

Con `gridSize` e `blockSize` si definisce:

- Numero totale di thread per un kernel.
- Il layout dei thread che si vuole utilizzare.

Come Eseguiamo il Codice in Parallelo sul Dispositivo?
- Sequenziale (non ottimale): ```cpp kernel_name<<<1, 1>>>(args); // 1 blocco, 1 thread per blocco```
- Parallelo: ```cpp kernel_name<<<256, 64>>>(args); // 256 blocchi, 64 thread per blocco ```


== Qualificatori di Funzione in CUDA

I qualificatori di funzione in CUDA sono essenziali per specificare dove una funzione verrà eseguita e da dove può essere chiamata.

#figure(image("images/_page39_2.1.png"))

```cpp
__global__ void kernelFunction(int data, int size);
```

- Funzione kernel (eseguita sulla GPU, chiamabile solo dalla CPU).

```cpp
__device__ int deviceHelper(int x);
```

- Funzione device (eseguita sulla GPU, chiamabile solo dalla GPU).

```cpp
__host__int hostFunction(int x);
```

- Funzione host (eseguibile su CPU).

=== Combinazione dei qualificatori host e device

In CUDA, combinando `__host__` e `__device__`, una funzione può essere eseguita sia sulla CPU che sulla GPU.

```cpp
__host__ __device__ int hostDeviceFunction(int x);
```

Permette di scrivere una sola volta funzioni che possono essere utilizzate in entrambi i contesti.

== Kernel CUDA: Regole e Comportamento
+ #text(weight: "bold", fill: green)[Esclusivamente Memoria Device] ( `__global__` e `__device__` )
  - Accesso consentito solo alla memoria della GPU. Niente puntatori verso la memoria host.
+ #text(weight: "bold", fill: green)[Ritorno void] ( `__global__` )
  - I kernel non restituiscono valori direttamente. La comunicazione con l'host avviene tramite la memoria.
+ #text(weight: "bold", fill: green)[Nessun supporto per argomenti variabili] ( `__global__` e `__device__` )
  - I kernel non possono avere un numero variabile di argomenti.
+ #text(weight: "bold", fill: green)[Nessun supporto per variabili statiche locali] ( `__global__` e `__device__` )
  - Tutte le variabili devono essere passate come argomenti o allocate dinamicamente.
+ #text(weight: "bold", fill: green)[Nessun supporto per puntatori a funzione] ( `__global__` e `__device__` )
  - Non è possibile utilizzare puntatori a funzione all'interno di un kernel.
+ #text(weight: "bold", fill: green)[Comportamento asincrono] ( `__global__` )
  - I kernel vengono lanciati in modo asincrono rispetto al codice host, salvo sincronizzazioni esplicite.

== Configurazioni di un Kernel CUDA

=== Combinazioni di Griglia 1D (Esempi)

La configurazione di griglia e blocchi può essere 1D, 2D o 3D (9 combinazioni in totale), permettendo una mappatura efficiente (ed intuitiva) su array, matrici o dati volumetrici.



```cpp
// 1D Grid, 1D Block
dim3 gridSize(4);
dim3 blockSize(8);
kernel_name<<<gridSize, blockSize>>>(args);
// 1D Grid, 2D Block
dim3 gridSize(4);
dim3 blockSize(8, 4);
kernel_name<<<gridSize, blockSize>>>(args);
// 1D Grid, 3D Block
dim3 gridSize(4);
dim3 blockSize(8, 4, 2);
kernel_name<<<gridSize, blockSize>>>(args);
```

Ottimale per problemi con dati strutturati linearmente, come l'elaborazione di *vettori* o *stringhe*, dove ogni thread può lavorare su una porzione contigua dei dati.

Nota: L'efficienza di una configurazione dipende da vari fattori come la dimensione dei dati, l'architettura della GPU e la natura del problema.


=== Combinazioni di Griglia 2D (Esempi)

La configurazione di griglia e blocchi può essere 1D, 2D o 3D (9 combinazioni in totale), permettendo una mappatura efficiente (ed intuitiva) su array, matrici o dati volumetrici.



```cpp
// 2D Grid, 1D Block
dim3 gridSize(4, 2);
dim3 blockSize(8);
kernel_name<<<gridSize, blockSize>>>(args);
// 2D Grid, 2D Block
dim3 gridSize(4, 2);
dim3 blockSize(8, 4);
kernel_name<<<gridSize, blockSize>>>(args);
// 2D Grid, 3D Block
dim3 gridSize(4, 2);
dim3 blockSize(8, 4, 2);
kernel_name<<<gridSize, blockSize>>>(args);
```


Ideale per problemi con dati strutturati in matrici o immagini, dove ogni thread può gestire un pixel o un elemento della matrice, sfruttando la vicinanza spaziale dei dati.

Nota: L'efficienza di una configurazione dipende da vari fattori come la dimensione dei dati, l'architettura della GPU e la natura del problema.


=== Combinazioni di Griglia 3D (Esempi)

La configurazione di griglia e blocchi può essere 1D, 2D o 3D (9 combinazioni in totale), permettendo una mappatura efficiente (ed intuitiva) su array, matrici o dati volumetrici.



```cpp
// 3D Grid, 1D Block
dim3 gridSize(4, 2, 2);
dim3 blockSize(8);
kernel_name<<<gridSize, blockSize>>>(args);
// 3D Grid, 2D Block
dim3 gridSize(4, 2, 2);
dim3 blockSize(8, 4);
kernel_name<<<gridSize, blockSize>>>(args);
// 3D Grid, 3D Block
dim3 gridSize(4, 2, 2);
dim3 blockSize(8, 4, 2);
kernel_name<<<gridSize, blockSize>>>(args);
```


Ottimale per problemi con *dati volumetrici*, come simulazioni fisiche o rendering 3D, dove ogni thread può operare su un voxel o una porzione dello spazio 3D.


== Numero di Thread per Blocco

- Il *numero massimo* totale di thread per blocco è 1024 per la maggior parte delle GPU (compute capability $>= 2.x$).
- Un blocco può essere organizzato in 1, 2 o 3 dimensioni, ma ci sono limiti per ciascuna dimensione. Esempio:
  - $x: 1024 , y: 1024, z: 64$
- Il prodotto delle dimensioni x, y e z #underline[*non*] può superare 1024 (queste limitazioni potrebbero cambiare in futuro).

#image("images/_page44_2.1.png")

== Compute Capability (CC) - Limiti SM

- La *Compute Capability (CC)* di NVIDIA è un numero che identifica le caratteristiche e le capacità di una GPU NVIDIA in termini di funzionalità supportate e limiti hardware.
- È composta da *due numeri*: il numero principale indica la *generazione* dell'architettura, mentre il numero secondario indica *revisioni* e *miglioramenti* all'interno di quella generazione.



== Identificazione dei Thread in CUDA

=== Esempio Codice CUDA

```cpp
#include <cuda_runtime.h>
// Kernel
__global__ void kernel_name() {
  // Accesso alle variabili built-in
  int blockId_x = blockIdx.x, blockId_y = blockIdx.y, blockId_z = blockIdx.z;
  int threadId_x = threadIdx.x, threadId_y = threadIdx.y, threadId_z = threadIdx.z;
  int totalThreads_x = blockDim.x, totalThreads_y = blockDim.y, totalThreads_z = blockDim.z;
  int totalBlocks_x = gridDim.x, totalBlocks_y = gridDim.y, totalBlocks_z = gridDim.z;
  // Logica del kernel...
}
int main() {
  // Definizione delle dimensioni della griglia e del blocco (Caso 3D)
  dim3 gridDim(4, 4, 2); // 4x4x2 blocchi
  dim3 blockDim(8, 8, 4); // 8x8x4 thread per blocco
  // Lancio del kernel
  kernel_name<<<gridDim, blockDim>>>();
  // Resto del Programma
}
```

== Tecniche di Mapping e Dimensionamento

=== Somma di Array in CUDA

Il Problema: Vogliamo sommare due array elemento per elemento in parallelo utilizzando CUDA.

#figure(image("images/_page48_2.1.png"))

*Approccio Tradizionale (CPU)*

- Gli elementi degli array vengono elaborati in sequenza, uno alla volta.
- Questo approccio è inefficiente per array di grandi dimensioni.
- Utilizza solo un core della CPU, rallentando il processo.

*Approccio CUDA (GPU)*

- Gli elementi degli array vengono elaborati in parallelo.
- La GPU è progettata per eseguire calcoli paralleli su larga scala.
- Migliaia di core della GPU lavorano insieme,

== Confronto: Somma di Vettori in C vs CUDA C


#figure(
)[
  #codly(header: [#align(center)[*Codice C Standard*]])
  ```c
  void sumArraysOnHost(float A, float B, 
  float C, int N) {
    for (int idx = 0; idx < N; idx++)
    C[idx] = A[idx] + B[idx];
  }
  // Chiamata della funzione
  sumArraysOnHost(A, B, C, N);
  ```
]



*Caratteristiche*

- *Esecuzione*: Sequenziale
- *Iterazione*: Loop Esplicito
- *Indice*: Variabile di Loop (`idx`)
- *Scalabilità*: Limitata dalla CPU

*Vantaggi*

- Portabilità su qualsiasi sistema
- Facilità di debugging



#figure(
)[
  #codly(header: [#align(center)[*Codice CUDA C*]])
  ```cpp
  __global__ void sumArraysOnGPU(float A, float B, 
  float C, int N) {
    int idx = ? // Come accedere ai dati?
    if (idx < N) C[idx] = A[idx] + B[idx];
  }
  // Chiamata del kernel
  sumArraysOnGPU<<<gridDim,blockDim>>>(A, B, C, N);
  ```
]



*Caratteristiche*

- *Esecuzione*: Parallela
- *Iterazione*: Implicita (un thread per elemento)
- *Indice*: ?
- *Scalabilità*: Elevata (sfrutta molti core GPU)

*Vantaggi*

- Altamente parallelo
- Eccellenti prestazioni su grandi dataset
- Sfrutta la potenza di calcolo delle GPU

Come mappare gli indici dei thread agli elementi dell'array?

// sumArraysOnGPU<<<1,12>>>(A, B, C) Grid Problemi Principali Scalabilità limitata: Funziona solo per array di dimensione uguale o inferiore al numero massimo di thread per blocco (tipicamente 1024 su molte GPU). Memoria GPU Mancanza di generalizzazione: Questo approccio non si estende facilmente a problemi di dimensioni arbitrarie o a griglie multi-dimensionali. A[idx] Utilizzo inefficiente della GPU: Questo approccio attiva solo uno dei + multi-processori (SM) disponibili sulla GPU, non sfruttando a pieno il B[idx] parallelismo offerto (lo vedremo analizzando il modello di esecuzione CUDA). C[idx] 14.90 8.20 11.08 25.98 38.43 1.07 35.65 88.48 28.52 34.01 23.35 6.20 C[0] C[1] C[2] C[3] C[4] C[5] C[6] C[7] C[8] C[9] C[10] C[11]

idx = threadIdx.x OK! Ma..

Come mappare gli indici dei thread agli elementi dell'array?


#figure(image("images/_page61_2.1.png"))


*Proprietà chiave di questo approccio *

- *Copertura completa*: Tutti i 12 thread (3 blocchi x 4 thread per blocco) sono utilizzati per elaborare i 12 elementi degli array.
- *Mapping corretto*: Ogni thread è associato a un unico elemento degli array A, B e C.
- *Nessuna ripetizione*: L'indice idx, univoco per ogni thread, assicura che ogni elemento dell'array venga elaborato esattamente una volta, evitando ridondanze.
- *Parallelismo massimizzato*: La formula idx permette di sfruttare appieno il parallelismo della GPU, assegnando un compito specifico ad ogni thread disponibile.
- *Scalabilità*: Questa formula si adatta bene a dimensioni di array diverse, purché si adegui il numero di blocchi.
- *Bilanciamento del carico*: Il lavoro è distribuito uniformemente tra tutti i thread, garantendo un utilizzo efficiente delle risorse.
- *Accessi coalescenti*: I thread adiacenti in un blocco accedono a elementi di memoria adiacenti, favorendo accessi coalescenti e migliorando l'efficienza della memoria.


Quindi il codice CUDA sarà il seguente:

#figure(
)[
  #codly(header: [#align(center)[*Codice CUDA C*]])
  ```cpp
  __global__ void sumArraysOnGPU(float A, float B, 
  float C, int N) {
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
  }
  // Chiamata del kernel
  sumArraysOnGPU<<<gridDim,blockDim>>>(A, B, C, N);
  ```
]



== Identificazione dei Thread e Mapping dei Dati in CUDA

Le variabili di identificazione sono accessibili solo all'interno del kernel e permettono ai thread di conoscere la propria posizione all'interno della gerarchia e di adattare il proprio comportamento di conseguenza.

*Perché Identificare i Thread?*

- L'indice globale di un thread identifica #underline[univocamente] quale porzione di dati deve elaborare.
- Essenziale per gestire correttamente l'accesso alla memoria e coordinare l'esecuzione di algoritmi complessi.

=== Struttura dei Dati e Calcolo dell'Indice Globale

- Anche le strutture più complesse, come matrici (2D) o array tridimensionali (3D), vengono memorizzate come una sequenza di elementi contigui in memoria nella GPU, tipicamente organizzati in array lineari.
- Ogni thread elabora uno o più elementi in base al proprio indice globale.
- Esistono diversi metodi per calcolare l'indice globale di un thread (es. Metodo Lineare, Coordinate-based).
- Metodi diversi possono produrre mappature diverse tra thread e dati, influenzando prestazioni (come la coalescenza degli accessi in memoria) e la leggibilità del codice.

=== Calcolo dell'Indice Globale del Thread - Grid 1D, Block 1D


In CUDA, ogni thread ha un indice globale (global\_idx) che lo identifica nell'esecuzione del kernel. Il programmatore lo calcola usando l'indice del thread nel blocco e l'indice del blocco nella griglia.

#figure(image("images/_page_64_Figure_2.jpeg"))

#{
  set text(fill: dark_orange)
  `blockIdx.x * blockDim.x`
}



- Calcola l'offset di tutti i thread nei blocchi precedenti al blocco corrente.
- Moltiplicando `blockIdx.x` per `blockDim.x`, otteniamo il numero totale di thread che si trovano nei blocchi precedenti.

#{
  set text(fill: light_green)
  `threadIdx.x`
}

- Identifica la posizione del thread all'interno del blocco corrente.
- È l'indice del thread all'interno del blocco corrente, da 0 a `blockDim.x - 1`.

In questo esempio viene mostrato come calcolare l'indice globale per il thread `Th(1)` appartenente al blocco unidimensionale con indice `blockIdx.x = 2`

#figure(image("images/_page_65_Figure_2.jpeg"))


In questo esempio viene mostrato come calcolare l'indice globale per il thread `Th(3)` appartenente al blocco unidimensionale con indice `blockIdx.x = 1`

#figure(image("images/_page_66_Figure_2.jpeg"))


In questo esempio viene mostrato come calcolare l'indice globale per il thread `Th(1)` appartenente al blocco unidimensionale con indice `blockIdx.x = 0`

#figure(image("images/_page_67_Figure_2.jpeg"))

== Calcolo dell'Indice Globale del Thread - Grid 1D, Block 2D

#figure(image("images/_page_68_Figure_1.jpeg"))
#let custom_rose = rgb("#FE6290")
#let custom_purple = rgb("#C899E6")
#let deep_red = rgb("#CF130C")

- #{ set text(fill: custom_rose); `ix = threadIdx.x + blockIdx.x * blockDim.x` }: determina l'indice del thread lungo l'asse `x`, prendendo in considerazione la posizione nel blocco (#{ set text(fill: custom_rose);`threadIdx.x`}) e il numero di blocchi precedenti (#{ set text(fill: custom_rose); `blockIdx.x * blockDim.x`}).
- #{ set text(fill: custom_purple); `iy = threadIdx.y + blockIdx.y * blockDim.y`}: determina l'indice del thread lungo l'asse y, considerando sia la posizione locale (#{ set text(fill: custom_purple); `threadIdx.y`}) che i blocchi precedenti lungo `y` (#{ set text(fill: custom_purple); `blockIdx.y * blockDim.y`}).
- #{ set text(fill: deep_red); `global_idx =`} #{ set text(fill: custom_purple); `iy`} `*` #{ set text(fill: blue); `nx`} + #{ set text(fill: custom_rose); `ix`}: calcola l'indice globale sommando #{ set text(fill: custom_rose); `ix`} all'indice globale lungo `y`, dove #{ set text(fill: blue); `nx`} rappresenta il numero di thread per riga (in questo caso, #{ set text(fill: blue); `nx = gridDim.x * blockDim.x`}).

Seguono alcuni esempi.

#figure(image("images/_page_69_Figure_1.jpeg"))



#figure(image("images/_page_70_Figure_1.jpeg"))


== Calcolo dell'Indice Globale del Thread - Grid 2D, Block 2D

#figure(image("images/_page73_2.1.png"))



== Metodo Basato su Coordinate per Indici Globali in CUDA

*Caratteristiche del Metodo Basato su Coordinate*

- Calcola indici separati per ogni dimensione della griglia e dei blocchi.
- Riflette naturalmente la disposizione multidimensionale dei dati.
- Facilita la comprensione della posizione del thread nello spazio
- Richiede un passaggio aggiuntivo per combinare gli indici in un indice globale.

=== Calcolo degli Indici Coordinati


#figure(image("images/_page75_2.1.png"))



=== Esempio di Utilizzo (Caso 2D)

```cpp
__global__ void kernel2D(float data, int width, int height) {
  int x = blockIdx.x  blockDim.x + threadIdx.x;
  int y = blockIdx.y  blockDim.y + threadIdx.y;
  if (x < width && y < height) { // width e height si riferiscono alle dimensioni dell'array dati
  int idx = y  width + x;
  // Operazioni su data[global_idx]
  }
}
```

=== Come Calcolare la Dimensione della Griglia e del Blocco

*Approccio Generale*

- Definire manualmente prima la dimensione del blocco (numero di thread per blocco).
- Poi, calcolare automaticamente la dimensione della griglia in base ai dati e alla dimensione del blocco.

*Motivazioni*

- La *dimensione del blocco* è legata alle *caratteristiche hardware* della GPU e la natura del problema.
- La *dimensione della griglia* si adatta alla *dimensione del blocco* e al *volume dei dati* da processare.


#figure(
)[
  #codly(header: [#align(center)[*Calcolo delle Dimensioni (Caso 1D)*]])
  ```cpp
  int blockSize = 256; int dataSize = 1024; // Dimensione del blocco e dei dati
  dim3 blockDim(blockSize); dim3 gridDim((dataSize + blockSize - 1) / blockSize); 
  kernel_name<<<gridDim, blockDim>>>(args); // Lancio del kernel
  ```
]

*Spiegazione del Calcolo*

La formula `(dataSize + blockSize 1) / blockSize` assicura un numero sufficiente di blocchi per coprire tutti i dati, anche se `dataSize` non è un multiplo esatto di `blockSize`.
  - *Divisione semplice*: `dataSize / blockSize` fornisce il numero di blocchi completamente pieni.
  - Se ci sono dati residui che non riempiono un intero blocco, la divisione semplice li ignorerebbe.
  - Aggiungere `blockSize - 1` a `dataSize` "compensa" questi dati residui, includendo l'ultimo blocco parziale. Equivalente a calcolare la ceil della divisione.




=== Esempio 1 (Dati Residui): `dataSize = 1030`, `blockSize = 256`

#figure(
)[
  #codly(header: [#align(center)[*Calcolo delle Dimensioni (Caso 1D)*]])
  ```cpp
  int blockSize = 256; int dataSize = 1030; // Dimensione del blocco e dei dati
  dim3 blockDim(blockSize); dim3 gridDim((dataSize + blockSize - 1) / blockSize);
  kernel_name<<<gridDim, blockDim>>>(args); // Lancio del kernel
  ```
]


- *Divisione semplice*: $1030 \/ 256 = 4$ blocchi; *ignorerebbe* l'ultimo blocco parziale perché
#math.equation(block: true, numbering: none, 
  $
  256 * 4 = 1024 \
  1030-1024 = 6
  $ 
)
quindi 6 elementi residui ❌
- Con la formula  $(1030 + 256 - 1) \/ 256 = 1285 \/ 256 = 5$ blocchi; nessun elemento residuo ✅
- In questo caso, la divisione semplice avrebbe dato 4 blocchi, ma c'è un residuo di 6 elementi ($1030 mod 256 = 6$); la formula include anche il blocco parziale, quindi otteniamo 5 blocchi.


=== Esempio 2 (Multiplo Perfetto): `dataSize = 1024`, `blockSize = 256`
- *Divisione semplice*: $1024 \/ 256 = 4$ → copre esattamente 1024 elementi ✅
- Con la formula $(1024 + 256 - 1) \/ 256 = 1279 \/ 256 = 4$ blocchi ✅
- *Spiegazione*: $1279 \/ 256 = 4.996$, ma essendo divisione intera → 4 blocchi; aggiungere 255 non basta per raggiungere 1280 (soglia del 5° blocco), quindi non si arrotonda per eccesso.
- *Importante*: La formula funziona correttamente sia con residui che senza, garantendo sempre il numero esatto di blocchi necessari senza sprechi

=== Calcolo delle Dimensioni (Caso 2D)

```cpp
int blockSizeX = 16, blockSizeY = 16; // Dimensione del blocco
int dataSizeX = 1024, dataSizeY = 512; // Dimensione dei dati
dim3 blockDim(blockSizeX, blockSizeY); // Definizione del blocco 2D
dim3 gridDim( // Calcolo della griglia 2D
 (dataSizeX + blockSizeX - 1) / blockSizeX, // Numero di blocchi in X
 (dataSizeY + blockSizeY - 1) / blockSizeY // Numero di blocchi in Y
);
kernel_name<<<gridDim, blockDim>>>(args); // Lancio del kernel
```


=== Calcolo delle Dimensioni (Caso Generale 3D)

```cpp
int blockSizeX = 16, blockSizeY = 16, blockSizeZ = 16; // Dimensione del blocco
int dataSizeX = 1024, dataSizeY = 512, dataSizeZ = 256; // Dimensione dei dati
dim3 blockDim(blockSizeX, blockSizeY, blockSizeZ); // Definizione del blocco 3D
dim3 gridDim( // Calcolo della griglia 3D
 (dataSizeX + blockSizeX - 1) / blockSizeX, // Numero di blocchi in X
 (dataSizeY + blockSizeY - 1) / blockSizeY, // Numero di blocchi in Y
 (dataSizeZ + blockSizeZ - 1) / blockSizeZ // Numero di blocchi in Z
);
kernel_name<<<gridDim, blockDim>>>(args); // Lancio del kernel
```

== Analisi delle Prestazioni


=== Verifica del Kernel CUDA (Somma di Array)

Il controllo dei kernel CUDA mira a confermare l'affidabilità dei calcoli eseguiti sulla GPU.

```cpp
void checkResult(float *hostRef, float *gpuRef, const int N) {
  double epsilon = 1.0E-8;
  int match = 1;
  for (int i = 0; i < N; i++) {
    if (fabsf(hostRef[i] - gpuRef[i]) > epsilon)
    {
      match = 0;
      printf("Arrays do not match!\n");
      printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
      break;
    }
  }
  if (match) printf("Arrays match.\n\n");
}
```
- `hostRef`: risultati attesi dalla somma
- `gpuRef`: risultati calcolati dal kernel
- `fabsf`: funzione della libreria C che calcola il valore assoluto di un numero in virgola mobile a precisione singola

*Suggerimenti per la Verifica (basic)*

- Verifica ogni elemento degli array per assicurarsi che i risultati del kernel corrispondano ai valori attesi.
- Usa una piccola tolleranza (`epsilon`) per confronti in virgola mobile, in quanto ci possono essere errori di arrotondamento legate alla natura delle rappresentazioni numeriche nei computer.
- *(Alternativa) Configurazione `<<< 1, 1>>>`*:
  - Forza l'esecuzione del kernel con un solo blocco e un thread.
  - Emula un'implementazione sequenziale.

=== Gestione degli Errori in CUDA

#{ set text(fill: light_green, weight: "bold", size: 12pt); "Problema"}

- *Asincronicità*: molte chiamate CUDA sono #underline[asincrone], rendendo difficile associare un errore alla specifica chiamata che lo ha causato.
- *Complessità di Debugging*: l'errore può emergere in una parte del programma diversa e lontana dal punto in cui è stato generato, rendendo l'individuazione della causa complicata.
- *Gestione Manuale*: controllare ogni chiamata CUDA manualmente è tedioso e soggetto a errori.

#figure(
)[
  #codly(header: [#align(center)[*Macro CHECK*]])
  ```cpp
  // Fornisce file, riga, codice e descrizione dell'errore.
  #define CHECK(call){
    const cudaError_t error = call;
    if (error != cudaSuccess){
      printf("Error: %s:%d, ", __FILE__, __LINE__);
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
      exit(1);
    }
  }
  ```
]


#figure(
)[
  #codly(header: [#align(center)[*Esempi di Utilizzo*]])
  ```cpp
  CHECK(cudaMalloc(&d_input, size)); // Allocazione
  CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
  kernel_function <<<numBlocks, blockSize >>>(argument list); // Lancia il kernel
  CHECK(cudaGetLastError()); // Primo controllo: errori di lancio del kernel
  CHECK(cudaDeviceSynchronize()); // Secondo controllo: errori durante l'esecuzione del kernel. Usare solo in DEBUG (Overhead di performance!)
  ```
]


=== Profiling delle Prestazioni in CUDA

#{ set text(fill: light_green, weight: "bold"); "Introduzione al Profiling"}

- Misurare e ottimizzare le prestazioni è fondamentale per garantire l'efficienza del codice.
- Il profiling permette di #underline[misurare le prestazioni], #underline[analizzare l'uso delle risorse] e #underline[individuare possibili ottimizzazioni].

#{ set text(fill: light_green, weight: "bold"); "Importanza della Misurazione del Tempo"}

- *Identificazione dei Colli di Bottiglia*: Individuare le sezioni di codice che limitano le prestazioni (un'implementazione naïve raramente è ottimale)
- *Analisi degli Effetti delle Modifiche*: valutare l'impatto delle modifiche sul tempo di esecuzione.
- *Confronto tra Implementazioni*: valutare le prestazioni tra diverse strategie di implementazione.
- *Analisi del Bilanciamento Carico/Calcolo*: verificare se il carico di lavoro è distribuito in modo efficiente tra thread, blocchi e host-device.

=== Metodi Principali

+ *Timer CPU*: Semplice e diretto, utilizza funzioni di sistema per ottenere il tempo di esecuzione.
+ *NVIDIA Profiler* (deprecato): Strumento da riga di comando per analizzare attività di CPU e GPU.
+ *NVIDIA Nsight Systems e Nsight Compute*: Strumenti moderni avanzati per analisi approfondita e ottimizzazione a livello di sistema e kernel.

==== Timer CPU
- Timer eseguito dall'host, misura il tempo *wall-clock* visto dalla CPU.
- Soluzione semplice e pratica basata su funzioni di sistema standard.
- *Può misurare qualsiasi operazione*: kernel GPU, trasferimenti memoria, codice CPU.
- Il tempo include anche gli overhead: lancio dei kernel, sincronizzazione, latenze di comunicazione.
- Per operazioni GPU asincrone (kernel), richiede sincronizzazione esplicita.

#figure(
)[
  #codly(header: [#align(center)[*Funzione del Timer della CPU*]])
  ```cpp
  #include <time.h>
  double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec  1.e-9);
    }
  ```
]



- La funzione utilizza `timespec_get()` per ottenere il tempo corrente del sistema.
- Restituisce il tempo in secondi, combinando secondi e nanosecondi.
- Precisione teorica fino al *nanosecondo*.



#figure(
)[
  #codly(header: [#align(center)[*Utilizzo Per Misurare un Kernel CUDA*]])
  ```cpp
  double iStart = cpuSecond(); // Registra il tempo di inizio 
  kernel_name<<<grid, block>>>(argument list); // Lancia il kernel CUDA
  cudaDeviceSynchronize(); // Attende il completamento del kernel
  double iElaps = cpuSecond() iStart; // Calcola il tempo trascorso
  ```
]

#rect(
  fill: rgb("#F3F3F3"),
  width: 100%,
  // height: 20em,
  inset: 1em,
  radius: 0.5em,
  [
  - La chiamata a ```cpp cudaDeviceSynchronize()``` è cruciale per assicurare che tutto il lavoro sulla GPU sia completato prima di misurare il tempo finale. Questo è necessario poiché le chiamate ai kernel CUDA sono *asincrone* rispetto all'host (senza rifletterebbe solo il tempo di lancio del kernel).
  - Il tempo misurato include sia l'*esecuzione* sia l'*overhead* di lancio e sincronizzazione.
  ]
)



#grid(
  columns: (1fr, 1fr),
  rows: auto,
  gutter: 1em,
  // Pro column
  rect(
    fill: rgb("#E5F0E1"),
    width: 100%,
    height: 16em,
    inset: 1em,
    radius: 0.5em,
    [
      #text(size: 1.2em, weight: "bold")[Pro]
      
      - Facile da implementare e utilizzare.
      - Non richiede librerie CUDA *specifiche* per il timing.
      - Funziona su *qualsiasi sistema* con supporto CUDA.
      - Efficace per *kernel lunghi* e *misure approssimative*.
    ]
  ),
  // Contro column
  rect(
    fill: rgb("#F4CCCC"),
    width: 100%,
    height: 16em,
    inset: 1em,
    radius: 0.5em,
    [
      #text(size: 1.2em, weight: "bold")[Contro]
      
      - Impreciso per kernel molto brevi ($< 1 \ms$).
      - Include *overhead* non relativo all'esecuzione del kernel (es., sistema operativo, utilizzo CPU, etc.).
      - Non fornisce dettagli sulle *fasi interne* del kernel.
      - Precisione influenzata dal *carico dell'host*.
    ]
  )
)

#figure(
)[
  #codly(header: [#align(center)[*Somma di due array*]])
  ```cpp
  int main(int argc, char **argv)
  {
    printf("%s Starting...\n", argv[0]);

    // Configurazione del device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); // Ottiene le proprietà del device
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev)); // Seleziona il device CUDA da utilizzare

    // Dimensione dei vettori
    int nElem = 1 << 24; // 2^24 elementi (16M) - bit shifting
    printf("Vector size %d\n", nElem);

    // Allocazione della memoria host
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes); // Alloca memoria per il vettore A su host
    h_B = (float *)malloc(nBytes); // Alloca memoria per il vettore B su host
    hostRef = (float *)malloc(nBytes); // Alloca memoria per il risultato calcolato su host
    gpuRef = (float *)malloc(nBytes); // Alloca memoria per il risultato calcolato su GPU

    // Inizializzazione dei dati su host
    double iStart, iElaps;
    iStart = cpuSecond();
    initialData(h_A, nElem); // Inizializza il vettore A
    initialData(h_B, nElem); // Inizializza il vettore B
    iElaps = cpuSecond() - iStart;
    printf("Data initialization time: %f sec\n", iElaps);
    memset(hostRef, 0, nBytes); // Inizializza a zero il vettore risultato su host
    memset(gpuRef, 0, nBytes); // Inizializza a zero il vettore risultato della GPU
    
    // Somma dei vettori su host per verifica
    iStart = cpuSecond();
    sumArraysOnHost(h_A, h_B, hostRef, nElem); // Calcola la somma su CPU per confronto
    iElaps = cpuSecond() - iStart;
    printf("sumArraysOnHost elapsed %f sec\n", iElaps);
    // Allocazione della memoria su device
    float *d_A, *d_B, *d_C;
  ```
]
#codly(header: [#align(center)[*Somma di due array*]], skips: ((1, 40), ))
  ```cpp
    CHECK(cudaMalloc((float**)&d_A, nBytes)); // Alloca memoria per A su GPU
    CHECK(cudaMalloc((float**)&d_B, nBytes)); // Alloca memoria per B su GPU
    CHECK(cudaMalloc((float**)&d_C, nBytes)); // Alloca memoria per il risultato su GPU
    // Copia dei dati dall'host al device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice)); // Copia A su GPU
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice)); // Copia B su GPU
    
    // Configurazione del kernel
    dim3 block(1024); // Dimensione del blocco: 1024 threads
    dim3 grid((nElem + block.x - 1) / block.x); // Calcola il numero di blocchi necessari
    
    // Esecuzione del kernel su device
    iStart = cpuSecond();
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem); // Lancia il kernel CUDA
    CHECK(cudaDeviceSynchronize()); // Attende il completamento del kernel
    iElaps = cpuSecond() - iStart;
    printf("sumArraysOnGPU <<<%d, %d>>> elapsed %f sec\n", grid.x, block.x, iElaps);
    
    // Copia dei risultati dal device all'host
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost)); // Copia dalla GPU
    
    // Verifica dei risultati
    checkResult(hostRef, gpuRef, nElem); // Confronta i risultati di CPU e GPU
    
    // Liberazione della memoria su device
    CHECK(cudaFree(d_A)); // Libera la memoria di A su GPU
    CHECK(cudaFree(d_B)); // Libera la memoria di B su GPU
    CHECK(cudaFree(d_C)); // Libera la memoria del risultato su GPU
    
    // Liberazione della memoria su host
    free(h_A); // Libera la memoria di A su host
    free(h_B); // Libera la memoria di B su host
    free(hostRef); // Libera la memoria del risultato CPU su host
    free(gpuRef); // Libera la memoria del risultato GPU su host
    return 0;}
  ```




#rect(
  fill: rgb("#F3F3F3"),
  //width: 100%,
  // height: 20em,
  inset: 1em,
  radius: 0.5em,
  [
  ```cpp int dev = 0```
  ]
)


-  Rappresenta l'*indice della GPU NVIDIA* che si intende utilizzare.
- `0` solitamente si riferisce al primo dispositivo CUDA disponibile nel sistema

  #{ set text(fill: light_green, weight: "bold"); "Alternative e Pratiche Comuni"}
  + Selezione del dispositivo tramite argomenti:\
      ```cpp if (argc > 1) dev = atoi(argv[1]);```
  + Utilizzo di variabili d'ambiente:
      ```cpp
      char* deviceIndex = getenv("CUDA_VISIBLE_DEVICES");
      if (deviceIndex) dev=atoi(deviceIndex);
    ```

#rect(
  fill: rgb("#F3F3F3"),
  //width: 100%,
  // height: 20em,
  inset: 1em,
  radius: 0.5em,
  [
  ```cpp cudaSetDevice(dev)```
  ]
)
- Imposta il dispositivo CUDA attivo per le operazioni successive
- Assicura che tutte le allocazioni e le operazioni CUDA successive utilizzino questo dispositivo specifico.

#rect(
  fill: rgb("#F3F3F3"),
  //width: 100%,
  // height: 20em,
  inset: 1em,
  radius: 0.5em,
  [
  ```cpp initialData(h_A, nElem); // Inizializza il vettore A```\
  ```cpp initialData(h_B, nElem); // Inizializza il vettore B```
  ]
)
- Inizializza un array di float con valori casuali (compresi fra 0 e 25.5)
    ```cpp
    void initialData(float *ip, int size){
      // Genera dati casuali
      time_t t;
      srand((unsigned int) time(&t));
      for (int i = 0; i < size; i++){
      ip[i] = (float)(rand() & 0xFF) / 10.0f;
      }
    }
    ```


*Compilazione con nvcc*

```sh
nvcc array_sum.cu -o array_sum
```

*Esecuzione e Risultato*

```sh
./array_sum Starting...
Using Device 0: NVIDIA GeForce RTX 3090
Vector size 16777216
Data initialization time: 0.425670 sec
sumArraysOnHost elapsed 0.033285 sec
sumArraysOnGPU <<<16384, 1024>>> elapsed 0.000329 sec
Arrays match.
```

*WorkStation*

#grid(
  columns: (1fr, 1fr),
  rows: auto,
  gutter: 1em,
  // Pro column
  rect(
    fill: rgb("#DEE9F4"),
    width: 100%,
    // height: 6em,
    inset: 1em,
    radius: 0.5em,
    [
      Intel Core i9-10920X (CPU)
      - Cores: 12 fisici (24 Threads)
      - Base Clock: 3.50 GHz
    ]
  ),
  // Contro column
  rect(
    fill: rgb("#E5F0E1"),
    width: 100%,
    // height: 6em,
    inset: 1em,
    radius: 0.5em,
    [
      NVIDIA GeForce RTX 3090 (GPU)
      - CUDA Cores: 10,496
      - Base Clock: 1.40 GHz
    ]
  )
)


#text(weight: "bold", fill: light_green)[Accesso ai Dati]

- *Efficienza della GPU*: La GPU esegue l'operazione circa 101 volte più velocemente della CPU ($0.033285 \/ 0.000329 tilde.triple 101$)
- *Overhead di Inizializzazione*: L'inizializzazione dei dati ($0.425670 s$) richiede circa 13 volte più tempo dell'elaborazione CPU.
- *Latenza vs Throughput*: Nonostante la CPU abbia una frequenza di clock più alta, la GPU supera significativamente le prestazioni grazie al massiccio parallelismo.



*Laptop*

#grid(
  columns: (1fr, 1fr),
  rows: auto,
  gutter: 1em,
  // Pro column
  rect(
    fill: rgb("#DEE9F4"),
    width: 100%,
    // height: 6em,
    inset: 1em,
    radius: 0.5em,
    [
      Intel(R) Core(TM) i7-11800H
      - Cores: 8 fisici (16 Threads)
      - Base Clock: 2.30 GHz
    ]
  ),
  // Contro column
  rect(
    fill: rgb("#E5F0E1"),
    width: 100%,
    // height: 6em,
    inset: 1em,
    radius: 0.5em,
    [
      NVIDIA GeForce RTX 3070 (GPU)
      - CUDA Cores: 5,120
      - Base Clock: 1.50 GHz
    ]
  )
)



#text(weight: "bold", fill: light_green)[Accesso ai Dati]

- *Efficienza della GPU*: La GPU esegue l'operazione circa 60 volte più velocemente della CPU ($0.039411 \/ 0.000650 tilde.triple 60$)
- *Overhead di Inizializzazione*: L'inizializzazione dei dati ($0.439789 s$) richiede circa 11 volte più tempo dell'elaborazione CPU.
- *Latenza vs Throughput*: Nonostante la CPU abbia una frequenza di clock più alta, la GPU supera significativamente le prestazioni grazie al massiccio parallelismo.
- *Confronto fra GPU*: Per questa operazione, la NVIDIA GeForce RTX 3090 è circa 1.97 volte più veloce della RTX 3070, con un tempo di esecuzione di 0.000329 secondi rispetto a $0.000650$ secondi.


==== Metodo 2: NVIDIA Profiler `[5.0 <= Compute Capability < 8.0]`

Dalla CUDA 5.0 è disponibile `nvprof`, uno strumento da riga di comando per raccogliere informazioni sull'attività di CPU e GPU dell'applicazione, inclusi kernel, trasferimenti di memoria e chiamate all'API CUDA.

#link("https://docs.nvidia.com/cuda/profiler-users-guide/")[Documentazione online]

```sh
$ nvprof [nvprof_args] <application> [application_args]
```

Ulteriori informazioni sulle opzioni di `nvprof` possono essere trovate utilizzando il seguente comando:

```sh
$ nvprof --help
```
Output integrabile in Visual Profiler con: ```sh nvprof -o file.nvvp ./app```

Nel nostro esempio:

```sh
$ nvprof ./array_sum
```

#rect(
  fill: rgb("#F3F3F3"),
  width: 100%,
  // height: 20em,
  inset: 1em,
  radius: 0.5em,
  [
    *Nota*

  - `nvprof` #underline[non è supportato] su dispositivi con Compute Capability ≥ 8.0 (Ampere+).
  - Per queste GPU, si consiglia di utilizzare *NVIDIA Nsight Systems* per il tracing della CPU/GPU, e *NVIDIA Nsight Compute* per il profiling della GPU.
  - Ancora disponibile su ambienti come Google Colab (GPU NVIDIA Tesla T4 Compute Capability: 7.5).
  ]
)


#figure(image("images/_page_17_2.1.png"))

==== NVIDIA Nsight Systems
#text(weight: "bold", fill: light_green)[Cos'è?] (#link("https://developer.nvidia.com/nsight-systems")[Documentazione Online])

- Strumento avanzato di *profilazione* e *analisi* delle prestazioni a livello di sistema.
- Offre una *visione globale dell'applicazione*, inclusi CPU, GPU e interazioni con il sistema.
- Permette di
  - Identificare *colli di bottiglia* nelle prestazioni.
  - Analizzare l'*overhead* delle chiamate API.
  - Esaminare le operazioni di *input/output*.
  - *Ottimizzare* il flusso di lavoro dell'applicazione.

#text(weight: "bold", fill: light_green)[Caratteristiche Chiave]

- *Visualizzazione grafica* delle timeline di esecuzione.
- *Analisi* delle chiamate API CUDA e sincronizzazioni.
- *Monitoraggio* dell'utilizzo di memoria e cache.
- *Supporto* per sistemi multi-thread e multi-GPU.


#rect(
  fill: rgb("#F3F3F3"),
  width: 100%,
  // height: 20em,
  inset: 1em,
  radius: 0.5em,
  [
    #green_heading("Output e Analisi")

  - Genera report dettagliati in vari formati (HTML, SQLite).
  - Fornisce grafici interattivi per visualizzare l'esecuzione nel tempo.
  - Permette di zoomare e navigare attraverso diverse sezioni dell'esecuzione.
  - Evidenzia automaticamente aree di potenziale ottimizzazione.
  ]
)


#green_heading("Come si usa?")

```sh 
nsys profile --stats=true ./array\_sum
```

Avvia il profiler e produce un'analisi dettagliata (non disponibile su Google Colab per GPU Tesla T4).

//#figure(image("images/_page_17_2.1.jpeg"))

*Timeline View*

#figure(image("images/_page_18_2.1.jpeg"))

*CUDA Summary (API/Kernels/MemOps)*

#figure(image("images/_page_100_2.1.png"))

#green_heading("Analisi del profiling (prima colonna)")

- *Gestione memoria domina*:
  - Allocazione (`cudaMalloc`): 55%
  - Trasferimenti (`cudaMemcpy` - operazioni di memoria): 20%
- *Esecuzione kernel GPU* trascurabile: 0.0% (244,222μs)
- *Operazioni ausiliarie* minime:
  - `cudaFree`: 3%
  - `cudaDeviceSynchronize`: ~0%
- *Conclusione*: Prestazioni limitate dalla gestione memoria, non dal calcolo GPU. Ottimizzazione dovrebbe concentrarsi su riduzione allocazioni e trasferimenti dati.


#rect(
  fill: rgb("#F3F3F3"),
  width: 100%,
  // height: 20em,
  inset: 1em,
  radius: 0.5em,
  [
    #align(center)[#blue_heading("Timer CPU (~329 μs)") vs #green_heading("Nsight Systems (244 μs)")]
    

    *Precisione*

    - #green_t("Nsight Systems"): Misurazioni precise, direttamente dalla GPU.
    - #blue_t("Timer CPU"): Meno preciso, include overhead extra (lancio, sincronizzazione).

    *Contesto*

    - #green_t("Nsight Systems"): Visione isolata del tempo di esecuzione del kernel GPU.
    - #blue_t("Timer CPU"): Include attività di sistema e altri processi, meno preciso.

    *Affidabilità*

    - #green_t("Nsight Systems"): Misurazioni stabili, meno influenzate da fattori esterni.
    - #blue_t("Timer CPU"): Vulnerabile alle fluttuazioni del sistema, meno affidabile.

    *Implicazioni per lo sviluppo*

    - #green_t("Nsight Systems"): Ottimizzazioni critiche, analisi approfondite, profiling accurato.
    - #blue_t("Timer CPU"): Stime approssimative nelle fasi iniziali, non per analisi dettagliate.
  ]
)


==== NVIDIA Nsight Compute

#green_heading("Cos'è?") (#link("https://developer.nvidia.com/nsight-compute")[Documentazione Online])

- Strumento di *profilazione* e *analisi* approfondita #underline[per singoli kernel CUDA].
- Fornisce *metriche dettagliate e mirate* alle prestazioni a livello di kernel.
- Permette di:
  - *Analizzare* l'utilizzo delle risorse GPU.
  - Identificare *colli di bottiglia* nei kernel.
  - Offre *report dettagliati* che possono essere utilizzati per ottimizzare il codice a livello di kernel.

#green_heading("Caratteristiche chiave")

- *Analisi* dettagliata delle metriche hardware per ogni kernel.
- *Visualizzazione grafica* dell'utilizzo della memoria.
- *Confronto* side-by-side di diverse esecuzioni dei kernel.
- *Suggerimenti automatici* per l'ottimizzazione.

#rect(
  fill: rgb("#F3F3F3"),
  width: 100%,
  // height: 20em,
  inset: 1em,
  radius: 0.5em,
  [
    #green_heading("Output e Analisi")

    - Genera report dettagliati in formato GUI o CLI.
    - Fornisce grafici e tabelle per visualizzare l'utilizzo delle risorse.
    - Permette l'analisi riga per riga del codice sorgente in relazione alle metriche.
    - Offre raccomandazioni specifiche per l'ottimizzazione basate sui dati raccolti.
  ]
)


#green_heading("Come si usa?")

```sh
ncu --set full -o test_report ./array_sum
```
- `-o test_report` è necessario per generare file per la visualizzazione grafica.
- avvia il profiler Nsight Compute e fornisce un'analisi dettagliata delle prestazioni dei kernel CUDA.


Utilizzando NVIDIA Nsight Compute, si può esaminare il tempo di esecuzione del kernel, evidenziando dettagli cruciali sull'uso della memoria e delle unità di calcolo.

#figure(image("images/_page_23_2.1.jpeg"))

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  // Box sinistro
  block(
    //fill: rgb("#F5F9E8"),
    stroke: 1pt + light_green,
    radius: 0.8em,
    inset: 1.5em,
    width: 100%,
    height: 21em
  )[
    #text(fill: light_green, weight: "bold")[Tempo di esecuzione del kernel]
    
    - 242,85 µs
    
    #v(0.5em)
    
    #text(fill: light_green, weight: "bold")[Throughput (specifico per l'esecuzione del kernel)]
    
    - *Compute (SM)*: 15,10% - Basso utilizzo delle unità di calcolo
    - *Memoria*: 89,62% - Alto utilizzo della banda di memoria
    - *Nota*: _Questi valori si riferiscono all'efficienza interna del kernel, non alle operazioni cudaMalloc/cudaMemcpy viste in Nsight Systems._
  ],
  // Box destro
  block(
    fill: my_gray,
    stroke: 1pt + rgb("#CCCCCC"),
    radius: 0.8em,
    inset: 1.5em,
    width: 100%,
    height: 21em
  )[
    #text(fill: light_green, weight: "bold")[Considerazioni]
    
    - Il kernel stesso è *memory-bound*, un aspetto non evidente dall'analisi di Nsight Systems.
    - Nsight Compute rivela che anche all'interno del kernel l'accesso alla memoria è il *collo di bottiglia*.
    - L'ottimizzazione dovrebbe considerare sia le *operazioni di memoria* a livello API (viste in Nsight Systems) che il *pattern di accesso alla memoria* all'interno del kernel (evidenziato da Nsight Compute).
  ]
)



=== Nvidia Nsight Systems vs. Compute

#green_heading("In Sintesi")

- *Nsight Systems* è uno strumento di analisi delle prestazioni a livello di sistema per identificare i colli di bottiglia delle prestazioni in #underline[tutto il sistema], inclusa la CPU, la GPU e altri componenti hardware.
- *Nsight Compute* è uno strumento di analisi e debug delle prestazioni a #underline[livello di kernel] per ottimizzare le prestazioni e l'efficienza di singoli kernel CUDA.

#figure(image("images/_page_25_2.1.png"))





=== Ottimizzazione della Gestione della Memoria in CUDA

- I trasferimenti di dati tra host e device attraverso il bus PCIe rappresentano un collo di bottiglia.
- *Allocazione sulla GPU*: L'allocazione di memoria sulla GPU è un'operazione relativamente lenta.

#rect(
  fill: rgb("#F3F3F3"),
  width: 100%,
  // height: 20em,
  inset: 1em,
  radius: 0.5em,
  [
    #green_heading("Best Practice")

    *Minimizzare i Trasferimenti di Memoria*

    - I trasferimenti di dati tra host e device hanno #underline[un'alta latenza].
    - Raggruppare i dati in buffer più grandi per #underline[ridurre i trasferimenti] e #underline[sfruttare la larghezza di banda].

    *Allocazione e Deallocazione Efficiente*

    - L'allocazione di memoria sulla GPU tramite `cudaMalloc` è un'operazione relativamente lenta.
    - Allocare la memoria una volta all'inizio dell'applicazione e riutilizzarla quando possibile.
    - Liberare la memoria con `cudaFree` quando non serve più, per evitare leak e sprechi di risorse.

    *Sfruttare la Shared Memory (vedremo in seguito)*

    - La shared memory è una memoria on-chip a bassa latenza accessibile a tutti i thread di un blocco.
    - Utilizzare la shared memory per i dati frequentemente acceduti e condivisi tra i thread di un blocco per ridurre l'accesso alla memoria globale più lenta.
  ]
)



== Applicazione Pratiche


=== Operazioni su Matrici in CUDA

- Dalla grafica 3D all'intelligenza artificiale, le *operazioni su matrici* sono il cuore di molti algoritmi. CUDA ci permette di eseguire queste operazioni in modo incredibilmente veloce, sfruttando la potenza delle GPU.
- In CUDA, come in molti altri contesti di programmazione, le matrici sono tipicamente memorizzate in *modo lineare* nella memoria globale utilizzando un approccio "_*row-major*_" (riga per riga).

#figure(image("images/_page_27_2.1.png"))

#figure(image("images/_page_28_2.1.jpeg"))

*Obiettivo*: Realizzare in CUDA la somma parallela di due matrici A e B, salvando il risultato in una matrice C.


#figure(image("images/_page_29_2.1.jpeg"))

*Mapping degli Indici*

Nell'elaborazione di matrici con CUDA, è fondamentale definire come i *thread vengono mappati agli elementi* della matrice. Questo processo di mapping incide #underline[direttamente] sulle prestazioni dell'algoritmo.

#block(
    //fill: rgb("#F5F9E8"),
    stroke: 1pt + light_green,
    radius: 0.8em,
    inset: 1.5em,
    width: 100%,
    height: 21em
  )[
    #text(fill: light_green, weight: "bold")[Problema Generale]

    Le matrici vengono linearizzate in memoria, quindi ogni elemento della matrice 2D
    deve essere mappato a un indice lineare: ```cpp idx = i * width + j``` , dove `width` è
    il numero di colonne della matrice e `(i, j)` sono le coordinate dell'elemento.

    
    #text(fill: light_green, weight: "bold")[Impatto della Configurazione]

    La configurazione scelta per la griglia e i blocchi (1D o 2D) influenza *come i thread sono associati agli elementi della matrice*.
      - Una configurazione adeguata permette a ogni thread di gestire *porzioni ben definite* dei dati.
      - Una configurazione non ottimale può portare a inefficienze, come thread che gestiscono intere colonne o righe della matrice, oppure che elaborano dati in modo non bilanciato.
  ]

*Suddivisione della Matrice*

Come possiamo suddividere questa matrice per eseguire il calcolo in parallelo? Cosa bisogna garantire?

#figure(image("images/_page_113_2.1.png", width: 70%))

#block(
  fill: my_gray,
  stroke: 0pt + rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em
)[
  #text(fill: light_green, weight: "bold")[Suddivisione]
  
  - La matrice può essere suddivisa in sottoblocchi di *dimensioni arbitrarie*.
  - La scelta delle dimensioni dei blocchi #underline[influenza] le *prestazioni*.


  #text(fill: light_green, weight: "bold")[Cosa Garantire]
  - *Copertura completa* della matrice.
  - *Scalabilità* per diverse dimensioni di matrice.
  - *Coerenza dei risultati* con l'elaborazione sequenziale.
  - *Accesso efficiente* alla memoria (lo vedremo in seguito).
]

==== Suddivisione della matrice in Griglia 2D e Blocchi 2D

#figure(image("images/_page_32_2.1.png", width: 70%))

#block(
  fill: my_gray,
  stroke: 0pt + rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  breakable: false
  // height: 21em
)[
  #text(fill: light_green, weight: "bold")[Organizzazione della Griglia]
  
  - La matrice è divisa, in questo caso specifico, in *6 blocchi*, in una configurazione 2x3 (`gridDim.x = 2, gridDim.y = 3`)
  - Ogni blocco è di dimensione 4x2, ovvero *8 thread* (`blockDim.x = 4, blockDim.y = 2`)
- Ogni thread ha un *indice locale* `(x, y)` all'interno del blocco.
- Ogni thread *elabora un elemento* della matrice.
]

#block(
  fill: my_gray,
  stroke: 0pt + rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em
)[
  #align(center)[
    #text(fill: light_green, weight: "extrabold",)[#underline[Attenzione!]\ Ambiguità tra Coordinate delle Matrici e dei Thread in CUDA]
  ]
  - Matrici: $mono(a_(i j))$ segue la #underline[convenzione riga, colonna] (indice `i` per la riga e `j` per la colonna).
    - *Esempio: $mono(a_(31))$* - la prima coordinata `i` rappresenta la riga (3) mentre la seconda coordinata `j` rappresenta la colonna (1).
  - *Thread/Blocco CUDA: `Th(x,y)/Block(x,y)`* utilizza una convenzione basata su #underline[coordinata cartesiane], con *`x,y`* riferiti alla posizione all'interno del blocco/griglia.
    - *Esempio: `Th(3,1)`* - la prima coordinata *`x`* rappresenta la posizione lungo l'asse $x$ (3), mentre la seconda coordinata *`y`* rappresenta la posizione lunga l'asse $y$ (1).
]

#green_heading("Come calcolo l'indice globale?")

Scegliamo un metodo di mapping, ad esempio quello *basato su coordinate* - ci concentriamo su quest'ultimo per la sua maggiore *intuitività*.


#figure(image("images/_page_36_2.1.png", width: 70%))

*Esempio di Mapping (#text(fill: red)[in rosso])*

Abbiamo che 
- `ix = threadIdx.x + blockIdx.x * blockDim.x`
- `iy = threadIdx.y + blockIdx.y * blockDim.y`
- `idx = iy * W + ix`

quindi:

+ *Indice x* nella matrice
  - `ix = 0 + 1 * 4 = 4`
+ *Indice y* nella matrice
  - `iy = 0 + 1 * 2 = 2`
+ *Indice lineare*
  - `idx = 2 * 8 + 4 = 20`
L'indice 20 corrisponde all'elemento $mono(a_(24))$.


#block(
  fill: my_gray,
  stroke: 0pt + rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em
)[
  #text(fill: light_green, weight: "bold")[Passi chiave]
  
  + *Validazione su Host*: Implementazione di una funzione di validazione `sumMatrixOnHost` in C.
  + *Kernel CUDA*: Definizione del kernel `sumMatrixOnGPU2D` che eseguirà la somma sulla GPU.
    - Viene configurata una griglia 2D di blocchi 2D per sfruttare il parallelismo massivo della GPU.
    - Ogni thread del kernel calcola il proprio *indice globale* dalle coordinate (`ix`, `iy`).
    - Ogni thread esegue l'operazione su un elemento delle matrici *`A`* e *`B`* e memorizza il risultato in *`C`*.

  + Configurazione:
    - Si definiscono le matrici su cui operare.
    - Si scelgono le dimensioni dei blocchi (`blockDim.x`, `blockDim.y`) per ottimizzare l'esecuzione sulle unità di calcolo della GPU.
    - Dimensioni della griglia in base alle dimensioni delle matrici e dei blocchi per coprire l'intera matrice: (```cpp dataSize + blockSize - 1) / blockSize``` (per ogni asse)
  + *Esecuzione*: Lanciare il kernel `sumMatrixOnGPU2D` sulla GPU con la configurazione definita.
]


*Confronto: Somma di Matrici in C vs CUDA C*

#figure(
)[
  #codly(header: [#align(center)[*Codice C Standard*]])
  ```c
  // Funzione host per la somma di matrici
  void sumMatrixOnHost(float MatA, float MatB, float MatC, int W, int H) {
    for (int i = 0; i < H; i++) { // Cicla su ogni riga
      for (int j = 0; j < W; j++) { // Cicla su ogni colonna
        int idx = i  W + j; // Calcola indice lineare
        MatC[idx] = MatA[idx] + MatB[idx]; // Somma elementi corrispondenti
      }
    }
  }
  ```
]

#figure(
)[
  #codly(header: [#align(center)[*Codice CUDA C*]])
  ```cpp
  // Kernel CUDA per la somma di matrici
  __global__ void sumMatrixOnGPU2D(float MatA, float MatB, float MatC, int W, int H) {
    unsigned int ix = threadIdx.x + blockIdx.x  blockDim.x; // Calcola indice x globale
    unsigned int iy = threadIdx.y + blockIdx.y  blockDim.y; // Calcola indice y globale
    if (ix < W && iy < H){ // Controlla limiti matrice
      unsigned int idx = iy  W + ix; // Calcola indice lineare
      MatC[idx] = MatA[idx] + MatB[idx]; // Somma elementi corrispondenti
    }
  }
  ```
]


#figure(
)[
  #codly(header: [#align(center)[*Somma di due matrici*]])
  ```cpp
  int main(int argc, char argv) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); // Ottiene le proprietà del device
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev)); // Seleziona il device CUDA

    // Imposta le dimensioni della matrice (16384 x 16384)
    int W = 1 << 14; // Soluzione migliore: passare le dimensioni tramite argomenti
    int H = 1 << 14;
    int size = W  H;
    int nBytes = size  sizeof(float);
    printf("Matrix size: W %d H %d\n", W, H);

    // Alloca la memoria host
    float h_A, h_B, hostRef, gpuRef;
    h_A = (float )malloc(nBytes); // Matrice A
    h_B = (float )malloc(nBytes); // Matrice B
    hostRef = (float )malloc(nBytes); // Risultato CPU
    gpuRef = (float )malloc(nBytes); // Risultato GPU

    // Inizializza i dati delle matrici (casualmente)
    initialData(h_A, size);
    initialData(h_B, size);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // Somma la matrice sulla CPU
    iStart = cpuSecond();
    sumMatrixOnHost(h_A, h_B, hostRef, W, H);
    iElaps = cpuSecond() - iStart;

    // Alloca la memoria del device
    float d_MatA, d_MatB, d_MatC;
    CHECK(cudaMalloc((void )&d_MatA, nBytes));
    CHECK(cudaMalloc((void )&d_MatB, nBytes));
    CHECK(cudaMalloc((void )&d_MatC, nBytes));

    // Trasferisce i dati dall'host al device
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));
  ```
]

#codly(header: [#align(center)[*Somma di due matrici*]], skips: ((1, 42), ))
```cpp 
    // Configura e invoca il kernel CUDA
    int block_dimx = 32; // Potrebbe assumere valori diversi (es. 16, 64, 128..) 
    int block_dimy = 32; // Potrebbe assumere valori diversi (es. 16, 64, 128..) 
    dim3 block(block_dimx, block_dimy);
    dim3 grid((W + block.x - 1) / block.x, 
              (H + block.y - 1) / block.y);

    iStart = cpuSecond();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, W, H);
    CHECK(cudaDeviceSynchronize()); // Sincronizza per misurare il tempo correttamente
    iElaps = cpuSecond() - iStart;

    // Copia il risultato del kernel dal device all'host
    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    // Verifica il risultato
    checkResult(hostRef, gpuRef, size);

    // continue...
```


*Griglia 2D e Blocchi 2D - Confronto fra Diverse Configurazioni*

#figure(image("images/_page_125_2.1.png"))

*Osservazioni*

- Tutte le configurazioni GPU offrono un *miglioramento* rispetto alla CPU.
- Miglioramento drastico passando da *`(1,1)`* a dimensioni di blocco maggiori.
- Le configurazioni con *più blocchi e thread* mostrano miglioramenti drammatici, con speedup superiori a *`131x`*.
- Le differenze tra le configurazioni *`(16,16)`* e *`(32,32)`* sono relativamente piccole, suggerendo una *saturazione* dell'utilizzo delle risorse GPU.
- Esiste un punto di ottimizzazione oltre il quale ulteriori aumenti nella dimensione o nel numero dei blocchi non producono miglioramenti significativi.

#block(
  fill: my_gray,
  stroke: 0pt + rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  breakable: false
  // height: 21em
)[
  #text(fill: light_green, weight: "bold")[Perché il blocco di dimensioni `(1,1)` è inefficiente?]
- *Overhead di gestione*: Lanciare tanti blocchi singoli crea un enorme overhead di scheduling e gestione per la GPU.
- *Mancato sfruttamento della località*: I thread non sono raggruppati in modo da sfruttare efficientemente la memoria cache e la memoria condivisa dei blocchi.
- *Inefficienza nell'utilizzo dei warp*: Le GPU operano su gruppi di thread chiamati _warp_ (tipicamente 32 thread). 
  Con un thread per blocco, la maggior parte delle unità di elaborazione in ogni warp rimane inutilizzata.
]


#block(
  fill: my_gray,
  stroke: 0pt + rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  breakable: false
  // height: 21em
)[
  #text(fill: light_green, weight: "bold")[Analisi Dettagliata da Nsight Compute - `(1,1)` vs `(16,16)`]
  #grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    - *Utilizzo della memoria (Memory [%])*
      - *Blocco 2D `(1,1)`*: 10,49%
      - *Blocco 2D `(16,16)`*: 94,28%
    
    - *Throughput di memoria:*
      - *Blocco 2D `(1,1)`*: 14,42 GB/s
      - *Blocco 2D `(16,16)`*: 860,79 GB/s
  ],
  [
    - *SM Busy:*
      - *Blocco 2D `(1,1)`* 5,64%
      - *Blocco 2D `(16,16)`*: 10,3%
    
    - *Occupancy:*
      - *Blocco 2D `(1,1)`*: 12,94
      - *Blocco 2D `(16,16)`*: 66,41
  ]
  )
  #set list(marker: [‣])
    - *SM Busy*: La configurazione `(16,16)` raddoppia l'utilizzo degli SM, 
      migliorando l'efficienza di calcolo.

    - *Occupancy Risorse*: Aumento di 5,1 volte nell'occupancy, indicando un 
      uso molto più efficiente delle risorse disponibili.

    - *Utilizzo della memoria*: Miglioramento drammatico nell'utilizzo della 
      larghezza di banda nel caso `(16,16)`, ottimizzando gli accessi alla 
      memoria.

    - *Throughput di memoria*: Aumento di circa 60 volte, principale fattore 
      del boost di performance complessivo.
]

==== Suddivisione della Matrice in Griglia 1D e Blocchi 1D

#figure(image("images/_page_128_2.1.png", width: 70%))


#block(
  fill: my_gray,
  stroke: 0pt + rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em
)[
  #text(fill: light_green, weight: "bold")[Organizzazione della Griglia]
  
  - La matrice è divisa, in questo caso specifico, in *4 blocchi*, in una configurazione 1D (`gridDim.x = 4`)
  - Ogni blocco ha configurazione 1D e contiene 2 thread (`blockDim.x = 2`)
  - Ogni thread ha un indice locale (`x`) all'interno del blocco
  - L'indice di mapping si calcola per ogni thread utilizzando gli indici del blocco e quelli locali lungo l'asse x
  
    ```cpp idx = ix ```
  - Ogni thread *elabora una colonna* della matrice (#underline[parallelismo limitato])
]

==== Confronto Kernel CUDA per la Somma fra Matrici


#codly(header: [#align(center)[*Griglia 2D e Blocchi 2D*]])
```cpp 
// Kernel CUDA per la somma di matrici
__global__ void sumMatrixOnGPU2D(float MatA, float MatB, float MatC, int W, int H) 
{
  unsigned int ix = threadIdx.x + blockIdx.x  blockDim.x; // Calcola indice x globale
  unsigned int iy = threadIdx.y + blockIdx.y  blockDim.y; // Calcola indice y globale
  if (ix < W && iy < H){ // Controlla limiti matrice
    unsigned int idx = iy  W + ix; // Calcola indice lineare
    MatC[idx] = MatA[idx] + MatB[idx]; // Somma elementi corrispondenti
  }
}
```

#codly(header: [#align(center)[*Griglia 1D e Blocchi 1D*]])
```cpp 
__global__ void sumMatrixOnGPU1D(float MatA, float MatB, float MatC, int W, int H) {
  unsigned int ix = threadIdx.x + blockIdx.x  blockDim.x; // Calcola indice x globale
  if (ix < W ) { // Controlla limiti matrice lungo l'asse x
    for (int iy = 0; iy < H; iy++) { // Scorre lungo l'asse y
      unsigned int idx = iy  W + ix; // Calcola indice lineare
      MatC[idx] = MatA[idx] + MatB[idx]; // Somma elementi corrispondenti
    }
  }
}
```

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  // Box sinistro
  block(
      fill: my_gray,
      stroke: 0pt + rgb("#F3F3F3"),
      radius: 0.8em,
      inset: 1.5em,
      width: 100%,
      breakable: false,
      height: 15em
    )[
      #align(center)[
        #text(fill: light_green, weight: "bold")[Griglia 2D & Blocchi 2D]
      ]
      - *Mappatura diretta*: Ogni thread gestisce un solo elemento 
        della matrice, sfruttando la natura bidimensionale del
        problema.
      - *Maggiore parallelismo*: Permette di sfruttare al massimo il 
        parallelismo offerto dalla GPU, con un thread per ogni
        elemento.

    ],
  //Box destro
  block(
    fill: my_gray,
    stroke: 0pt + rgb("#F3F3F3"),
    radius: 0.8em,
    inset: 1.5em,
    width: 100%,
    breakable: false,
    height: 15em
  )[
    #align(center)[
      #text(fill: light_green, weight: "bold")[Griglia 1D & Blocchi 1D]
    ]
    - *Minore parallelismo*: Ogni thread gestisce una colonna 
      intera, limitando il parallelismo a livello di riga.
    - *Loop interno*: Il ciclo for introduce un'inefficienza, poiché ogni thread
      deve iterare su tutti gli elementi della sua colonna
  ]
)

#block(
  breakable: false
)[
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1.5em,
    // Box sinistro
    text(size: 10pt, fill: nvidia-green, weight: "bold")[
        NVIDIA Nsight Compute\*
    ],
    //Box destro
    align(center)[
      #text(size: 10pt, weight: "bold")[Dim. Matrice] 
      #text(size: 10pt)[(`16384,16384`)]
    ]
  )

  #block(
    stroke: (paint: nvidia-green, thickness: 1pt),
    radius: 10pt,
    inset: 20pt,
  )[

    #table(
      columns: (auto, auto, auto, auto, auto),
      align: (center, center, center, center, center),
      stroke: none,
      inset: 4pt,
      
      
      [*Dim. Griglia*], 
      [*Dim. Blocco*], 
      [*Runtime (ms)*], 
      [*Speedup vs CPU*], 
      [*Device*],
      
      [`--`], [`--`], [`516,08` (TimerCPU)], [], 
      text(fill: device-blue)[i9-10920X (CPU)],
      
      table.cell(colspan: 5, align: left)[#line(length: 100%, stroke: (dash: "dashed"))],
      
      [`4096`], [`4`], [`24,49*`], [`21,07x`], 
      text(fill: nvidia-green)[RTX 3090 (GPU)],
      
      [`1024`], [`16`], [`7,69*`], [`67,11x`], 
      text(fill: nvidia-green)[RTX 3090 (GPU)],
      
      [`512`], [`32`], [`7,22*`], [`71,48x`], 
      text(fill: nvidia-green)[RTX 3090 (GPU)],
      
      [`256`], [`64`], [`7,22*`], [`71,48x`], 
      text(fill: nvidia-green)[RTX 3090 (GPU)],
      
      [`128`], [`128`], [`7,20*`], [*`71,68x`*], 
      text(fill: nvidia-green)[RTX 3090 (GPU)],
      
      [`64`], [`256`], [`7,22*`], [`71,48x`], 
      text(fill: nvidia-green)[RTX 3090 (GPU)],
    
    )
  ]
]

#block(
    fill: my_gray,
    stroke: 0pt + rgb("#F3F3F3"),
    radius: 0.8em,
    inset: 1.5em,
    width: 100%,
    breakable: false
    // height: 21em
  )[
    *Osservazioni*
    - Prestazioni relativamente *uniformi* con *`Dim.Blocco > 16`*,
      con tempi di esecuzione tra *`7,20`* e *`7,69`* ms.
    - Lo speedup rispetto alla CPU varia da *`67,11x`* a *`71,68x`*, inferiore all'approccio griglia 2D e blocchi 2D ma comunque significativo.
    - Mentre abbiamo *parallelismo lungo l'asse x* (ogni thread gestisce una colonna), l'*elaborazione lungo l'asse y è sequenziale*. Questo riduce significativamente il parallelismo effettivo rispetto agli approcci 2D.
  ]

#image("images/_page_132_2.1.png")

#block(
  fill: my_gray,
  stroke: 0pt + rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  breakable: false
  // height: 21em
)[
  - *Limite dei blocchi*: Limite massimo dei 65535 thread per blocco sull'asse *y* (vedi compute capability GPU)
  - *Necessità di adattamento*: Per gestire matrici con un numero di righe superiori è necessario modificare
    la configurazione per suddividere il lavoro in modo diverso.
]

#image("images/_page_134_2.1.png")

#block(
  fill: my_gray,
  stroke: 0pt + rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  breakable: false
  // height: 21em
)[
  #align(center)[
    #text(fill: light_green, weight: "bold")[Tendenza nelle Prestazioni]
  ]
  
  - *Griglia 1D, Blocchi 1D*: Le prestazioni peggiorano significativamente all'aumentare
    del numero di righe delle matrici (da *71,48x* a *1,40x* di speedup rispetto alla CPU).
  - *Griglia 2D, Blocchi 2D*: Mantiene prestazioni costanti e elevate 
    (speedup tra *131,09x* e *137,99x* rispetto alla CPU) per tutte le dimensioni di matrice.
  - *Caso estremo*: Per la matrice (*`1048576,256`*), l'approccio 1D1D diventa di poco superiore
    ad un approccio sequenziale (*1,40x*), mentre il 2D2D mantiene un alto speedup (*131,09x*).
]


==== Suddivisione della Matrice in Griglia 1D e Blocchi 2D

#figure(image("images/_page_136_2.1.png", width: 70%))

#block(
fill: my_gray,
stroke: 0pt + rgb("#F3F3F3"),
radius: 0.8em,
inset: 1.5em,
width: 100%,
breakable: false
// height: 21em
)[
  #text(fill: light_green, weight: "bold")[Organizzazione della Griglia]
  - La matrice è divisa, in questo caso specifico, in *8 blocchi*, in una configurazione 1D (`gridDim.x = 8`).
  - Ogni blocco ha configurazione 2D e contiene 6 thread (`blockDim.x = 1,blockDim.y = 6`) - degenere
  - Ogni thread ha un indice locale `(0,y)` all'interno del blocco.
  - Ogni thread *elabora un elemento* della matrice (sempre?)
  - L'indice di mapping si calcola per ogni thread *combinando gli indici del blocco e quelli locali*
  
    ```cpp idx = iy * W + ix```
]

#block(
  fill: my_gray,
  stroke: 0pt + rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  breakable: false
  // height: 21em
)[
  #text(fill: light_green, weight: "bold")[Accesso in Memoria]
  - Le matrici sono memorizzate in ordine _"row-major"_. Questa configurazione *non sfrutta la località spaziale* dei dati in memoria.
  - I thread in ogni blocco accedono ad elementi di memoria *non contigui* (*estremamente inefficiente* - lo vedremo)
  - *Transazioni multiple* di memoria invece di una singola transazione
    ottimizzata (maggiore latenza e ridotto throughput)
]

==== Confronto Kernel CUDA per la Somma fra Matrici - 2D2D 1D2D
#codly(header: [#align(center)[*Griglia 2D e Blocchi 2D (Esempio Precedente)*]])
```cpp 
// Kernel CUDA per la somma di matrici
__global__ void sumMatrixOnGPU2D(float MatA, float MatB, float MatC, int W, int H) {
  unsigned int ix = threadIdx.x + blockIdx.x  blockDim.x; // Calcola indice x globale
  unsigned int iy = threadIdx.y + blockIdx.y  blockDim.y; // Calcola indice y globale
  if (ix < W && iy < H){ // Controlla limiti matrice
    unsigned int idx = iy  W + ix; // Calcola indice lineare
    MatC[idx] = MatA[idx] + MatB[idx]; // Somma elementi corrispondenti
  }
}
```
#codly(header: [#align(center)[*Griglia 1D e Blocchi 2D (con una dimensione degenere)*]])
```cpp
__global__ void sumMatrixOnGPU1D2D(float MatA, float MatB, float MatC, int W, int H) {
  unsigned int ix = blockIdx.x; // Calcola indice x globale
  unsigned int iy = threadIdx.y; // Calcola indice y globale
  if (ix < W && iy < H) { // Controlla limiti matrice
    unsigned int idx = iy  W + ix; // Calcola indice lineare
    MatC[idx] = MatA[idx] + MatB[idx]; // Somma elementi corrispondenti
  }
}
```
// Griglia
#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  // Box sinistro
  block(
      fill: my_gray,
      stroke: 0pt + rgb("#F3F3F3"),
      radius: 0.8em,
      inset: 1.5em,
      width: 100%,
      breakable: false,
      height: 22em
    )[
      #align(center)[
        #text(fill: light_green, weight: "bold")[Griglia 2D & Blocchi 2D]
      ]
      - *Limiti Thread*: Max 1024 thread per blocco, vincolati dalla compute capability.
      - *Distribuzione*: 
        - Può distribuire i thread su entrambe le dimensioni. 
        - Divide sia righe che colonne in blocchi.
      - *Scalabilità*: Buona per matrici grandi, suddivide il lavoro uniformemente.

    ],
  //Box destro
  block(
    fill: my_gray,
    stroke: 0pt + rgb("#F3F3F3"),
    radius: 0.8em,
    inset: 1.5em,
    width: 100%,
    breakable: false,
    height: 22em
  )[
    #align(center)[
      #text(fill: light_green, weight: "bold")[Griglia 1D & Blocchi 1D (degenere)]
    ]
    - *Limiti Thread*: Max 1024 thread/colonna, vincolati dalla compute capability.
    - *Distribuzione*: 
      - Thread limitati a una singola dimensione (*y*) del blocco. 
      - Divide la matrice solo per colonne. 
      - Dimensione del blocco almeno pari al numero di righe (a meno di adattamenti al codice).
    - *Scalabilità*: 
      - Potenziali difficoltà con matrici con molte righe (>1024). 
      - Richiede adattamenti del codice (es. loop).
  ]
)

#block(
  fill: my_gray,
  stroke: 1pt + rgb("#000000"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  breakable: false,
  // height: 22em
)[
  *Nota*: Il primo kernel (2D-2D) è *più generale*: può gestire qualsiasi configurazione di griglia/blocchi. Il secondo (1D-2D) è una *versione specializzata*:

  *Esempio*: Configurando il lancio del primo kernel con ```cpp dim3 block (1, blockDim_y); dim3 grid (W, 1);``` le formule diventano:

  ```
  ix = threadIdx.x + blockIdx.x * blockDim.x → ix = 0 + blockIdx.x * 1 
  iy = threadIdx.y + blockIdx.y  blockDim.y → iy = threadIdx.y + 0
  ```
]

#block(
  breakable: false
)[
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1.5em,
    // Box sinistro
    text(size: 10pt, fill: nvidia-green, weight: "bold")[
        NVIDIA Nsight Compute\*
    ],
  )

  #block(
    stroke: (paint: nvidia-green, thickness: 1pt),
    radius: 10pt,
    inset: 20pt,
  )[

    #table(
      columns: (auto, auto, auto, auto, auto, auto),
      align: (center, center, center, center, center, center),
      stroke: none,
      inset: 3pt,
      
      [*Dim. Matrice*],
      [*Dim. Griglia*], 
      [*Dim. Blocco*], 
      [*Runtime (ms)*], 
      [*Speedup (vs CPU)*], 
      [*Device*],
      
      [`(512,512)`], [`--`], [`--`], [`0,505` (TimerCPU)], [], 
      text(fill: device-blue)[i9-10920X (CPU)],
      
      [`(512,4096)`], [`--`], [`--`], [`4,065` (TimerCPU)], [], 
      text(fill: device-blue)[i9-10920X (CPU)],
      
      [`(512,16384)`], [`--`], [`--`], [`16,12` (TimerCPU)], [], 
      text(fill: device-blue)[i9-10920X (CPU)],
      
      [`(1024,16384)`], [`--`], [`--`], [`33,92` (TimerCPU)], [], 
      text(fill: device-blue)[i9-10920X (CPU)],
      
      [`(2048,16384)`], [`--`], [`--`], [`64,41` (TimerCPU)], [], 
      text(fill: device-blue)[i9-10920X (CPU)],
      
      table.cell(colspan: 6, align: left)[#line(length: 100%, stroke: (dash: "dashed"))],
      
      table.cell(colspan: 6, align: left)[
        #text(fill: nvidia-green, weight: "bold")[Griglia 1D, Blocchi 2D (degenere)]
      ],
      
      [`(512,512)`], [`512`], [`(1,512)`], [`0,021*`], [`24,05x`], 
      text(fill: nvidia-green)[RTX 3090 (GPU)],
      
      [`(512,4096)`], [`4096`], [`(1,512)`], [`0,153*`], [`26,57x`], 
      text(fill: nvidia-green)[RTX 3090 (GPU)],
      
      [`(512,16384)`], [`16384`], [`(1,512)`], [`0,607*`], [`26,56x`], 
      text(fill: nvidia-green)[RTX 3090 (GPU)],
      
      [`(1024,16384)`], [`16384`], [`(1,512)`], 
      text(fill: red)[#sym.times], text(fill: red)[#sym.times], 
      text(fill: nvidia-green)[RTX 3090 (GPU)],
      
      [`(1024,16384)`], [`16384`], [`(1,1024)`], [`1,21*`], [`28,03x`], 
      text(fill: nvidia-green)[RTX 3090 (GPU)],
      
      [(#text(fill: red)[#underline[`2048`]]`,16384)`], [`16384`], 
      [`(1,` #text(fill: red)[#sym.times]`)`], 
      text(fill: red)[#sym.times], text(fill: red)[#sym.times], 
      text(fill: nvidia-green)[RTX 3090 (GPU)],
      
      table.cell(colspan: 6, align: left)[#line(length: 100%, stroke: (dash: "dashed"))],
      
      table.cell(colspan: 6, align: left)[
        #text(fill: nvidia-green, weight: "bold")[Griglia 2D, Blocchi 2D]
      ],
      
      [`(1024,16384)`], [`(512,32)`], [`(32,32)`], [`0,245*`], 
      [*`138,45x`*], 
      text(fill: nvidia-green)[RTX 3090 (GPU)],
    
    )
  ]
]

#block(
  fill: my_gray,
  stroke: 1pt + rgb("#000000"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  breakable: false,
  // height: 22em
)[
  *Note sulla configurazione `(2048,16384)`*
  - *Limite dei thread*: Limite massimo dei 1024 thread per blocco (vedi compute capability GPU)
  - *Necessità di adattamento*: Per gestire matrici così grandi con questa configurazione,
    sarebbe necessario modificare il codice per suddividere il lavoro in modo diverso. Come?
]

#block(
  fill: my_gray,
  stroke: 0pt + rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  breakable: false
  // height: 21em
)[
  #align(center)[
    #text(fill: light_green, weight: "bold")[Analisi Dettagliata da Nsight Compute (2D2D vs 1D2D)]
  ]
  #grid(
  columns: (1fr, 1fr),
  gutter: 2cm,
  [
    *Utilizzo della memoria (Memory [%])*
    - *1D2D:* 83,95%
    - *2D2D:* 89,40%

    *Throughput di memoria:*
    - *1D2D:* 164,20 GB/s
    - *2D2D:* 810,80 GB/s
  ],
  [
    *SM Busy:*
    - *1D2D:* 1,66%
    - *2D2D:* 11,28%

    *Cicli di stallo per istruzione:*
    - *1D2D:* 361,92 cicli
    - *2D2D:* 50,85 cicli
  ]
)

#v(0.5cm)

- *Accesso alla Memoria:* La versione 2D ottimizza gli accessi coalescenti, migliorando drasticamente il throughput di memoria.

- *Parallelismo:* Migliore distribuzione del carico di lavoro nella 2D, con maggiore utilizzo dei multiprocessori ed efficienza delle istruzioni.

- *Riduzione stalli:* La 2D minimizza i cicli di attesa per thread, migliorando l'efficienza complessiva.

- *Granularità:* La suddivisione del lavoro nella 2D permette una migliore sovrapposizione di calcolo e accessi memoria.
]


==== Suddivisione della Matrice in  Griglia 2D e Blocchi 1D

#figure(image("images/_page_62_2.1.jpeg", width: 70%))

#block(
  fill: my_gray,
  stroke: 0pt + rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  breakable: false
  // height: 21em
)[
  #text(fill: light_green, weight: "bold")[Organizzazione della Griglia]
  - La matrice è divisa, in questo caso specifico, in *24 blocchi*, in una configurazione 4x6 (`gridDim.x = 4, gridDim.y = 6`)
  - Ogni blocco è 1D di dimensione 2, ovvero *2 thread* (`blockDim.x = 2`)
  - Ogni thread ha un *indice locale* (`x`) all'interno del blocco
  - Ogni thread *elabora un elemento* della matrice.

  Esempio di Mapping (#text(fill: red, weight: "bold")[in rosso]):
  + *Indice x* nella matrice
    
    `ix = 1 + 2 * 2 = 5`

  + *Indice y* nella matrice
  
    `iy = 0 + 4 * 1 = 4`

  + *Indice lineare*

    `idx = iy * W + ix = 4 * 8 + 5 = 37`

  L'indice 37 corrisponde all'elemento $mono(a_(24))$.
]

==== Confronto Kernel CUDA per la Somma fra Matrici - 2D2D 2D1D

#codly(header: [#align(center)[*Griglia 2D e Blocchi 2D (Esempio Precedente)*]])
```cpp
// Kernel CUDA per la somma di matrici
__global__ void sumMatrixOnGPU2D(float MatA, float MatB, float MatC, int W, int H) {
  unsigned int ix = threadIdx.x + blockIdx.x  blockDim.x; // Calcola indice x globale
  unsigned int iy = threadIdx.y + blockIdx.y  blockDim.y; // Calcola indice y globale
  if (ix < W && iy < H){ // Controlla limiti matrice
    unsigned int idx = iy  W + ix; // Calcola indice lineare
    MatC[idx] = MatA[idx] + MatB[idx]; // Somma elementi corrispondenti
  }
}
```

#codly(header: [#align(center)[*Griglia 2D e Blocchi 1D*]])
```cpp
__global__ void sumMatrixOnGPU2D1D(float MatA, float MatB, float MatC, int W, int H) {
  unsigned int ix = threadIdx.x + blockIdx.x  blockDim.x; // Calcola indice x globale
  unsigned int iy = blockIdx.y; // Calcola indice y globale
  if (ix < W && iy < H){ // Controlla limiti matrice
    unsigned int idx = iy  W + ix; // Calcola indice lineare
    MatC[idx] = MatA[idx] + MatB[idx]; // Somma elementi corrispondenti
  } 
}
```


#block(
  fill: my_gray,
  stroke: 1pt + rgb("#000000"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  breakable: false,
  // height: 22em
)[
  *Nota*: Anche in questo caso il primo kernel (2D-2D) è *più generale* - 
  può gestire qualsiasi configurazione di griglia/blocchi. 
  
  Il secondo (2D-1D) è una *versione specializzata*:

  *Esempio*: Configurando il lancio del primo kernel con `cpp dim3 block (blockDim_x, 1); dim3 grid (gridDim_x, H);` le formule diventano:

  ``` 
  ix = threadIdx.x + blockIdx.x * blockDim.x → rimane invariata
  iy = threadIdx.y + blockIdx.y * 1 → iy = threadIdx.y * 0
  ```

  *Strategia inversa*: Qui usiamo threading solo su X e griglia su Y, mentre nel caso precedente (1D-2D) era il contrario.
]

#block(
  breakable: false,
)[
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1.5em,
    // Box sinistro
    text(size: 10pt, fill: nvidia-green, weight: "bold")[
        NVIDIA Nsight Compute\*
    ],
    //Box destro
    align(center)[
      #text(size: 10pt, weight: "bold")[Dim. Matrice] 
      #text(size: 10pt)[(`16384,16384`)]
    ]
  )

  #block(
    stroke: (paint: nvidia-green, thickness: 1pt),
    radius: 10pt,
    inset: 10pt,
  )[

    #table(
      columns: (auto, auto, auto, auto, auto),
      align: (center, center, center, center, center),
      stroke: none,
      inset: 4pt,
      
      [*Dim. Griglia*], 
      [*Dim. Blocco*], 
      [*Runtime (ms)*], 
      [*Speedup vs CPU*], 
      [*Device*],
      
      [`--`], [`--`], [`516,08` (TimerCPU)], [], 
      text(fill: device-blue)[i9-10920X (CPU)],
      
      table.cell(colspan: 5, align: left)[#line(length: 100%, stroke: (dash: "dashed"))],
      
      [`(1024,16384)`], [`16`], [`13,98*`], [`36,92x`], 
      text(fill: nvidia-green)[RTX 3090 (GPU)],
      
      [`(512,16384)`], [`32`], [`6,99*`], [`73,83x`], 
      text(fill: nvidia-green)[RTX 3090 (GPU)],
      
      [`(256,16384)`], [`64`], [`3,75*`], [*`137,62x`*], 
      text(fill: nvidia-green)[RTX 3090 (GPU)],
      
      [`(128,16384)`], [`128`], [`3,75*`], [*`137,62x`*], 
      text(fill: nvidia-green)[RTX 3090 (GPU)],
      
      [`(64,16384)`], [`256`], [`3,75*`], [*`137,62x`*], 
      text(fill: nvidia-green)[RTX 3090 (GPU)],
      
      [`(32,16384)`], [`512`], [`3,76*`], [`137,25x`], 
      text(fill: nvidia-green)[RTX 3090 (GPU)],
    
    )
  ]
]


#block(
  fill: my_gray,
  stroke: 0pt + rgb("#000000"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  breakable: false,
  // height: 22em
)[
  *Osservazioni*
  - Le migliori prestazioni si raggiungono con configurazioni a più di 64   
    thread per blocco, tutte con un tempo di esecuzione di *3,75 ms*.
  - Miglioramento significativo passando da 16 thread (*13,98 ms*) a 
    32 thread (*6,99 ms*), e ulteriore miglioramento fino a 64.
  - La configurazione con più thread per blocco permette un migliore utilizzo
    delle risorse hardware, risultando in prestazioni superiori 
    (*Suggerimento*: osservare analisi completa con Nsight Compute)
]



==== Confronto fra le Migliori Configurazioni di Blocchi e Griglie

#block(
  breakable: false
)[
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1.5em,
    // Box sinistro
    text(size: 10pt, fill: nvidia-green, weight: "bold")[
        NVIDIA Nsight Compute\*
    ],
    //Box destro
    align(center)[
      #text(size: 10pt, weight: "bold")[Dim. Matrice] 
      #text(size: 10pt)[(`16384,16384`)]
    ]
  )

  #block(
    stroke: (paint: nvidia-green, thickness: 1pt),
    radius: 10pt,
    inset: 20pt,
  )[

    #table(
      columns: (auto, auto, auto, auto, auto, auto),
      align: (center, center, center, center, center),
      stroke: none,
      inset: 4pt,
      
      [*Conf.*],
      [*Dim. Griglia*], 
      [*Dim. Blocco*], 
      [*Runtime (ms)*], 
      [*Speedup vs CPU*], 
      [*Device*],
      
      [`--`],[`--`], [`--`], [`516,08` (TimerCPU)], [], 
      text(fill: device-blue)[i9-10920X (CPU)],
      
      table.cell(colspan: 6, align: left)[#line(length: 100%, stroke: (dash: "dashed"))],
      
      [*1D1D*],[`128`], [`128`], [`7,20*`], [`71,68x`], 
      text(fill: nvidia-green)[RTX 3090 (GPU)],
      
      [*1D2D*],[`16384`], [`(1,` #text(fill: red)[#underline[16384]]) #text(fill: red)[(*NO!*)] ], [`--`], [`--`], 
      text(fill: nvidia-green)[RTX 3090 (GPU)],
      
      [*2D1D*],[`(256,16384)`], [`64`], [`3,75*`], [*`137,62x`*], 
      text(fill: nvidia-green)[RTX 3090 (GPU)],
      
      [*2D2D*],[`(1024,1024)`], [`(16,16)`], [`3,75*`], [*`137,62x`*], 
      text(fill: nvidia-green)[RTX 3090 (GPU)],
    
    )
  ]
]

#block(
  fill: my_gray,
  stroke: 0pt + rgb("#000000"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  breakable: false,
  // height: 22em
)[
  *Osservazioni*
  - L'approccio *Grid 1D* e *Blocchi 1D* mostra prestazioni generalmente inferiori,
    con uno speedup massimo di *`71,68x`* rispetto alla CPU (il loop per thread limita le prestazioni).
  - L'approccio *Grid 1D* e *Blocchi 2D* (degenere) non è in grado di gestire 
    queste dimensioni della matrice (righe > 1024) senza modifiche al codice. Ogni thread dovrebbe processare più elementi della matrice.
  - L'approccio *Grid 2D* e *Blocchi 1D* raggiunge prestazioni equivalenti, 
    ma sacrifica la semplicità concettuale del mapping diretto matrice #sym.arrow.l.r griglia/blocchi 2D.
  - L'approccio Grid 2D e Blocchi 2D offre le migliori prestazioni complessive,
    con uno speedup di 137,62x
  - La scelta dell'approccio ottimale dipende dalle caratteristiche specifiche 
    del problema, come le dimensioni della matrice, la struttura dei dati e le capacità dell'hardware.
]

== Immagini come Matrici Multidimensionali

#block(
    //fill: rgb("#F5F9E8"),
    stroke: 1pt + light_green,
    radius: 0.8em,
    inset: 1.5em,
    width: 100%,
    // height: 21em,
    breakable: false
  )[
    #green_heading("Struttura di Base")

    - Un'*immagine digitale* è una *griglia di pixel*.
    - Ogni pixel rappresenta il *colore* o *l'intensità* di un punto specifico nell'immagine.
    - Questa griglia può essere rappresentata matematicamente come una *matrice*.
  ]


#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  // Box sinistro
  [
    #figure(image("images/_page_148_2.1.jpeg"), caption: "Immagine a Colore (RGB)")

    - *Dimensioni*: Larghezza x Altezza x 3 (canali)
    - Ogni pixel è rappresentato da tre valori: 
      #text(fill: red)[Rosso], #text(fill: green)[Verde], 
      #text(fill: blue)[Blu] (RGB).
  ]
  ,
  //Box destro
  [
    #figure(image("images/_page_148_2.1_2.jpeg"), caption: "Immagine Grayscale")

    - *Dimensioni*: Larghezza x Altezza
    - Ogni elemento della matrice è un singolo valore di intensità `[0..255]`
  ]
  
)

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  // Box sinistro
  [
    #figure(image("images/_page_150_2.1_1.png"), caption: "Immagine a Colore (RGB)")

    - *Dimensioni*: Larghezza x Altezza x 3 (canali)
    - Ogni pixel è rappresentato da tre valori: 
      #text(fill: red)[Rosso], #text(fill: green)[Verde], 
      #text(fill: blue)[Blu] (RGB).
  ]
  ,
  //Box destro
  [
    #figure(image("images/_page_150_2.1_2.png"), caption: "Immagine Grayscale")

    - *Dimensioni*: Larghezza x Altezza
    - Ogni elemento della matrice è un singolo valore di intensità `[0..255]`
  ]
)



=== Memorizzazione Lineare di Immagini RGB in CUDA

- Per le immagini in *scala di grigi*, la memorizzazione in memoria globale è diretta e segue esattamente il principio *row-major* delle matrici classiche viste in precedenza.
- Per le immagini *RGB*, il principio di base rimane lo stesso, ma con una *complessità aggiuntiva* dovuta ai tre canali di colore (ogni pixel occupa 3 posizioni in memoria).


#block(
  fill: my_gray,
  stroke: 0pt + rgb("#000000"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  breakable: false,
  // height: 22em
)[
  *Approccio di Memorizzazione (Caso RGB)*

  Ci sono due approcci principali per memorizzare un'immagine RGB in modo lineare:

  + *Planar*:
    - Tutti i valori R, poi tutti i G, poi tutti i B
    #figure(image("images/_page_150_2.1.jpeg"))
  + *Interleaved* (#underline[più comune]):
  - I valori R, G, B per ogni pixel sono memorizzati consecutivamente
    #image("images/_page_151_2.1.png")
]



#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  // Box sinistro
  [
    #figure(image("images/_page_152_2.1_1.png"))

    #block(
      fill: my_gray,
      stroke: 0pt + rgb("#000000"),
      radius: 0.8em,
      inset: 1.5em,
      width: 100%,
      breakable: false,
      // height: 22em
    )[
      Per accedere a un pixel specifico `(i, j)`:
      - *Calcola l'indice di base:*

        ``` baseIndex = (i * width + j) * 3```
      - *Accesso ai canali:*
        - #text(fill: red, weight: "bold")[R]: `baseIndex`
        - #text(fill: green, weight: "bold")[G]: `baseIndex + 1`
        - #text(fill: blue, weight: "bold")[B]: `baseIndex + 2`
    ]
  ]
  ,
  //Box destro
  [
    #figure(image("images/_page_152_2.1_2.png", width: 92%))

    #block(
      fill: my_gray,
      stroke: 0pt + rgb("#000000"),
      radius: 0.8em,
      inset: 1.5em,
      width: 100%,
      breakable: false,
      // height: 22em
    )[
      - *Dimensioni*: Larghezza x Altezza
      - Ogni elemento della matrice è un singolo valore di intensità `[0..255]`
    ]
  ]
  

  
)



=== Parallelismo GPU nella Conversione RGB a Grayscale

#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
    #green_heading("Perché le GPU sono Ideali per l'Elaborazione delle Immagini")

    - *Struttura delle Immagini*
      - Le immagini sono composte da molti *pixel indipendenti.*
      - Ogni pixel può essere elaborato *separatamente.*
    - *Operazioni Uniformi*
      - La *stessa operazione* viene spesso applicata a tutti i pixel.
      - Perfetto per il paradigma *SIMD* (Single Instruction, Multiple Data).
]
#block(
  breakable: false,

)[
  #align(center)[*Esempio: Conversione RGB a Grayscale*]
  #image("images/_page_153_2.1.png")
  #align(center)[*Formula*: #text(fill: gray)[Gray] = 0.299#text(fill: red)[R] + 0.587#text(fill: green)[G] + 0.114#text(fill: blue)[B] (per pixel)]
]


==== Suddivisione dell'Immagine in Blocchi per l'Elaborazione GPU

#figure(image("images/_page_154_2.1.png", width: 70%))

#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  - L'elaborazione di immagini su GPU richiede la *suddivisione* del lavoro in *unità parallele.*
  - L'immagine viene divisa in una *griglia* di *blocchi*, ciascuno elaborato
    da un gruppo di thread.
    - *`gridDim`*: Numero di blocchi nella griglia.
    - *`blockDim`*: Numero di thread in ciascun blocco.
]

#block(
  fill: my_gray,
  stroke: 0pt + rgb("#000000"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  breakable: false,
  // height: 22em
)[
  #green_heading("Calcolo degli indici nel buffer RGB")

  ```cpp
  ix = threadIdx.x + blockIdx.x * blockDim.x
  iy = threadIdx.y + blockIdx.y * blockDim.y
  base_index = (iy * width + ix) * 3
  index_R = base_index
  index_G = base_index + 1
  index_B = base_index + 2
  ```
]

#block(
  fill: my_gray,
  stroke: 0pt + rgb("#000000"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  breakable: false,
  // height: 22em
)[
  #align(center)[
    #green_heading("Threads Oltre i Limiti dell'Immagine")
  ]

  - Le dimensioni dei blocchi sono tipicamente potenze di 2.
  - Le dimensioni delle immagini raramente sono multipli esatti di queste dimensioni dei blocchi.
  - Per coprire l'intera immagine, si lanciano spesso *più blocchi del necessario*, alcuni dei quali si estendono *oltre i bordi* dell'immagine.
  - I thread *che cadono fuori* dai limiti dell'immagine semplicemente *non eseguono* alcuna operazione.
]


=== Confronto: Conversione RGB a Grayscale in C vs CUDA C

#codly(header: [#align(center)[*Codice C Standard*]])
```c
// Funzione host per la conversione RGB->Gray
void rgbToGrayCPU(unsigned char *rgb, unsigned char *gray, int width, int height) {
  for (int y = 0; y < height; y++) { // Ciclo su tutte le righe dell'immagine
    for (int x = 0; x < width; x++) { // Ciclo su tutti i pixel di una riga
      int rgbOffset = (y * width + x) * 3; // Calcola l'offset per il pixel RGB
      int grayOffset = y * width + x; // Calcola l'offset per il pixel in scala di grigi
      unsigned char r = rgb[rgbOffset]; // Legge il valore rosso
      unsigned char g = rgb[rgbOffset + 1]; // Legge il valore verde
      unsigned char b = rgb[rgbOffset + 2]; // Legge il valore blu
      gray[grayOffset] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b); // RGB->Gray
    }
  }
}
```
#codly(header: [#align(center)[*Codice CUDA C*]])
```cpp
// Funzione kernel per la conversione RGB->Gray
__global__ void rgbToGrayGPU(unsigned char *d_rgb, unsigned char *d_gray, int width, int height) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x; // Calcola la coordinata x del pixel
  int iy = blockIdx.y * blockDim.y + threadIdx.y; // Calcola la coordinata y del pixel
  if (ix < width && iy < height) { // Controllo dei bordi: assicura che il thread sia dentro l'immagine
    int rgbOffset = (iy * width + ix) * 3; // Calcola l'offset per il pixel RGB
    int grayOffset = iy * width + ix; // Calcola l'offset per il pixel in scala di grigi
    unsigned char r = d_rgb[rgbOffset]; // Legge il valore rosso
    unsigned char g = d_rgb[rgbOffset + 1]; // Legge il valore verde
    unsigned char b = d_rgb[rgbOffset + 2]; // Legge il valore blu
    d_gray[grayOffset] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b); // RGB->Gray
  }
}
```


=== Conversione RGB a Grayscale in CUDA
#codly(header: [#align(center)[*Conversione RGB -> Grayscale*]])
```cpp
int main(int argc, char *argv) {
  if (argc != 2) {
    printf("Usage: %s <image_file>\n", argv[0]);
    return 1;
  }
  printf("%s Starting...\n", argv[0]);

  // Imposta il device CUDA
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev)); // Ottiene le proprietà del dispositivo CUDA
  CHECK(cudaSetDevice(0)); // Seleziona il dispositivo CUDA
  
  // Carica l'immagine usando "stb_image.h" e "stb_image_write.h"
  int width, height, channels;
  unsigned char *rgb = stbi_load(argv[1], &width, &height, &channels, 3);
  if (!rgb) {
    printf("Error loading image %s\n", argv[1]);
    return 1;
  }
  printf("Image loaded: %dx%d with %d channels\n", width, height, channels);

  // Alloca la memoria host per l'immagine in scala di grigi
  int imageSize = width * height;
  int rgbSize = imageSize * 3;
  unsigned char *h_gray = (unsigned char *)malloc(imageSize); // Alloca memoria per l'output GPU
  unsigned char *cpu_gray = (unsigned char *)malloc(imageSize); // Alloca memoria per l'output CPU

  // Converti l'immagine in scala di grigi sulla CPU
  rgbToGrayscaleCPU(rgb, cpu_gray, width, height);

  // Alloca la memoria del device
  unsigned char *d_rgb, *d_gray;
  CHECK(cudaMalloc((void *)&d_rgb, rgbSize)); // Alloca memoria GPU per l'immagine RGB
  CHECK(cudaMalloc((void *)&d_gray, imageSize)); // Alloca memoria GPU per l'output 

  // Trasferisce i dati dall'host al device
  CHECK(cudaMemcpy(d_rgb, rgb, rgbSize, cudaMemcpyHostToDevice));

  // Configura e invoca il kernel CUDA
  dim3 block(32, 32); // Dimensione del blocco: 32x32 thread (altre dimensioni possibili)
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  rgbToGrayscaleGPU<<<grid, block>>>(d_rgb, d_gray, width, height); // Lancia il kernel
  CHECK(cudaDeviceSynchronize()); // Aspetta il completamento del kernel

  // Copia il risultato del kernel dal device all'host
  CHECK(cudaMemcpy(h_gray, d_gray, imageSize, cudaMemcpyDeviceToHost));

  // Verifica il risultato
  bool match = true;
  for (int i = 0; i < imageSize; i++) {
    if (abs(cpu_gray[i] - h_gray[i]) > 1) { // Tollera piccole differenze di arrotondamento.
      match = false;
      printf("Mismatch at pixel %d: CPU %d, GPU %d\n", i, cpu_gray[i], h_gray[i]);
      break;
    }
  }
  if (match) printf("CPU and GPU results match.\n");

  // Salva l'immagine in scala di grigi
  stbi_write_png("output_gray.png", width, height, 1, h_gray, width);

  // Libera la memoria
  stbi_image_free(rgb);
  free(h_gray);
  free(cpu_gray);
  CHECK(cudaFree(d_rgb));
  CHECK(cudaFree(d_gray));

  // Resetta il device CUDA
  CHECK(cudaDeviceReset());
  return 0;
}
```

#block(
  fill: my_gray,
  stroke: 0pt + rgb("#000000"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  breakable: false,
  // height: 22em
)[
  *Nota:* vedi documentazione 
  #link("https://github.com/nothings/stb/blob/master/stb_image.h")[`stb_image.h`]
  e #link("https://github.com/nothings/stb/blob/master/stb_image_write.h")[`stb_image_write.h`].
]


=== Image Flipping con CUDA

#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  - L'image flipping è una tecnica di elaborazione delle immagini che
    *inverte l'ordine dei pixel lungo un asse* specifico per ciascun canale
    di colore, creando un *effetto specchio*.

    Il flipping può essere:
    - *Orizzontale:* Invertendo l'ordine dei pixel da sinistra a destra.
    - *Verticale:* Invertendo l'ordine dei pixel dall'alto verso il basso.
]

#figure(image("images/_page_161_2.1.png"))

#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Processo di Flipping in CUDA")
  - In CUDA, ogni thread è responsabile del calcolo e della gestione di un 
    singolo pixel dell'immagine.
    - Per un *flip orizzontale*, il thread calcola la nuova posizione speculare
      del pixel. Per un pixel inizialmente in posizione `(i, j)`, il thread calcola
      la nuova posizione come (`i, width -1 -j)`.
    - Per un *flip verticale*, la nuova posizione è calcolata come `(height -1 -i, j)`.
- Il thread *copia i valori* dei canali RGB del pixel originale nella nuova posizione calcolata.
]

#figure(image("images/_page_162_2.1.png"))

#codly(header: [#align(center)[*Flipping di un'Immagine*]])
```cpp
__global__ void cudaImageFlip(unsigned char* input, unsigned char* output, int width, int height, int channels, bool horizontal) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x; // Calcola la coordinata x del pixel
  int iy = blockIdx.y * blockDim.y + threadIdx.y; // Calcola la coordinata y del pixel
  if (ix < width && iy < height) { // Verifica se il pixel è all'interno dell'immagine
    int outputIdx;
    int inputIdx = (iy * width + ix) * channels;
    if (horizontal) {
      outputIdx = (iy * width + (width - 1 - ix)) * channels; // Indice flip orizzontale
      } else {
        outputIdx = ((height - 1 - iy) * width + ix) * channels; // Indice flip verticale
      }
      for (int c = 0; c < channels; ++c) {
        output[outputIdx + c] = input[inputIdx + c]; // Copia i valori nella nuova posizione
      }
    }
}
```


=== Image Blur con CUDA
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Introduzione all'Image Blurring")

  L'image blurring è una tecnica di elaborazione delle immagini che *riduce i dettagli* e le *variazioni di intensità*, creando un *effetto di sfocatura*. 

  Viene utilizzata per:

  - *Riduzione del rumore*: Attenuando le fluttuazioni casuali dei pixel.
  - *Enfasi degli oggetti*: Sfumando i dettagli irrilevanti e mettendo in 
    risalto gli elementi principali.
  - *Preprocessing per la Computer Vision*: Semplificando l'immagine per 
    facilitarne l'analisi da parte degli algoritmi.
]

#figure(image("images/_page_164_2.1.png"))

#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Concetto di Base")

  Il blurring si ottiene calcolando la *media dei valori di intensità* dei pixel
  vicini di ogni pixel dell'immagine originale. L'operazione può essere riassunta come segue:
  - *Patch di dimensioni NxN:* Una patch (o finestra) di dimensioni fisse scorre su ciascun pixel dell'immagine.
  - *Pixel centrale:* Ogni pixel di output è la media dei pixel nella patch che lo circondano.
  - *Esempio con patch 3x3:* Include il pixel centrale più gli 8 pixel che lo 
    circondano, formando una matrice
]
#figure(image("images/_page_164_2.1.jpeg"))

#figure(image("images/_page_173_2.1.jpeg"))

#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Caratteristiche Chiave del Kernel Blur")

  - *Mappatura Thread-Pixel:* Ogni thread è responsabile del calcolo
    di un singolo pixel nell'immagine di output.
  - *Gestione dei Bordi:* Controlli specifici assicurano che la finestra
    di blur rimanga entro i confini dell'immagine,
    evitando letture di memoria non valide ai margini.
  - *Parallelismo:* Il kernel sfrutta il parallelismo massiccio delle GPU,
    dato che il calcolo per ciascun pixel è indipendente dagli altri.
  - *Pattern di Accesso alla Memoria*: Ogni thread accede a un vicinato
    di pixel (la patch) che, a seconda della disposizione dei dati in memoria,
    può comportare accessi *non sempre sequenziali*.

]

#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Confronto con Kernel Precedenti")

  - *Complessità:* Rispetto a semplici kernel come *`vecAdd`* (addizione vettoriale)
    o *`rgbToGray`* (conversione in scala di grigi), questo kernel è più complesso
    a causa della necessità di gestire più pixel e calcoli per ogni thread.
  - *Accessi alla Memoria*: Ogni thread accede a più pixel rispetto a
    kernel semplici, aumentando la frequenza di accessi alla memoria globale.
  - *Scalabilità*: La dimensione della patch di blur (`BLUR_SIZE`) impatta
    direttamente la quantità di calcolo e gli accessi alla memoria. 
    Patch più grandi producono sfocature più intense ma richiedono più risorse
]

#codly(header: [#align(center)[*Image Blur con CUDA*]])
```cpp
#define BLUR_RADIUS 1 // Raggio del blur (1 significa una finestra 3x3)
__global__ void cudaImageBlur(unsigned char* input, unsigned char* output, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    int pixelSum = 0, pixelCount = 0;
    // Itera sulla finestra di blur
    for (int dy = -BLUR_RADIUS; dy <= BLUR_RADIUS; ++dy) {
      for (int dx = -BLUR_RADIUS; dx <= BLUR_RADIUS; ++dx) {
        int currentY = y + dy, currentX = x + dx;
        // Verifica se il pixel è all'interno dell'immagine
        if (currentY >= 0 && currentY < height && currentX >= 0 && currentX < width) {
          pixelSum += input[currentY * width + currentX];
          pixelCount ++;
        }
      }
    }
    // Calcola e scrive il valore medio del pixel
    output[y * width + x] = (unsigned char)(pixelSum / pixelCount);
  }
}
```

=== Introduzione alla Convoluzione 1D e 2D

#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Che cos'è la Convoluzione?")

  - Operazione matematica lineare *tra due funzioni*, segnale e kernel
    (#underline[fuorviante] spesso indicato come *filtro*).
  - Misura la *sovrapposizione* del filtro con il segnale mentre scorre su di esso.
  - Produce una nuova funzione (segnale di output) che rappresenta 
    le *caratteristiche estratte* dal segnale di input.
]

#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Convoluzione 1D")

  - Applicata a *dati unidimensionali* (segnali audio, serie temporali, sequenze di testo)
  - Il filtro è un vettore che *scorre* sul segnale
  - L'output ad ogni punto è la *somma dei prodotti elemento per elemento
    (prodotto scalare)* tra il filtro e la porzione dell'immagine sottostante
  - *Esempio*: Applicazione di un filtro di media mobile su un segnale audio per ridurre il rumore

  #green_heading("Convoluzione 2D")

  - Applicata a *dati bidimensionali* (es. immagini).
  - Il filtro è una matrice che *scorre* sull'immagine
  - L'output ad ogni pixel è la *somma dei prodotti elemento per elemento
    (prodotto scalare)* tra il filtro e la regione dell'immagine sottostante
  - *Esempio*:
    - (Image Blur caso particolare di convoluzione 2D. Perché?)
    - Fondamentale nelle *reti neurali convoluzionali (CNN)* per l'elaborazione di immagini
]

#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #align(center)[*Concetti aggiuntivi*]
  - *Padding:* consiste nell'aggiungere un *bordo di valori* (solitamente zeri)
    attorno all'input prima di applicare la convoluzione. Utile per controllare
  - *Stride*: definisce il *passo* con cui il kernel si sposta sull'input durante 
    la convoluzione. Uno stride maggiore comporta sia una riduzione delle dimensioni
    dell'output che un aumento della velocità di elaborazione
]


==== Esempio di Convoluzione 1D

#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #grid(
    columns: (1fr, 1fr),
    rows: auto,
    gutter: 1em,
    //Left box
    [
      #green_heading("Descrizione")
      - *Input (I)*: Array di 7 elementi (I[0]...I[6]).
      - *Filtro (F)*: Array di 5 elementi (F[0]...F[4]).
      - *Output (O)*: Array risultante dalla convoluzione di l con F.
    ]
    ,
    //Right Box
    [
      #green_heading("Formalmente")
      #math.equation(
        numbering: none,
        block: true, 
        $ O[i] = sum_(j=-r)^r F[j], I[i+j] $
      )
      $r$: raggio del filtro 1D
    ]
    
    
  )
]

#figure(image("images/_page_181_2.1.jpeg"))

==== Perché la Convoluzione si Adatta al Calcolo Parallelo

#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Indipendenza dei Calcoli")

  - Ogni elemento di output è calcolato *indipendentemente.*
  - Permette l'*elaborazione parallela.*

  #green_heading("Operazioni Uniformi")

  - Stesse operazioni ripetute *su diverse porzioni dei dati.*
  - Si allinea con l'architettura *SIMD.*

  #green_heading("Mapping Diretto Thread-Output")

  - *Ogni thread* può calcolare un elemento di output.
  - Semplifica la parallelizzazione del problema.
]

#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Implementazione Generica: Passi")
  - Un thread GPU *per ogni elemento* di output.
  - Ogni thread:
    - *Identifica* regione input corrispondente.
    - *Applica* il filtro e *calcola* risultato.
    - *Scrive* output.

  *Nota*: Questa è un'implementazione "#underline[naive]". Ottimizzazioni avanzate
  saranno trattate successivamente.
]

#codly(header: [#align(center)[*CUDA Convoluzione 1D: Soluzione (non ottimale)*]])
```cpp
__global__ void cudaConvolution1D(float* input, float* output, float* filter, int W, int
filterSize )
{
  int x = blockIdx.x * blockDim.x + threadIdx.x; // Indice globale del thread
  int radius = filterSize / 2; // Raggio del filtro (supponiamo filterSize dispari)
  if (x < W) // Verifica che il thread sia all’interno dei limiti dell’input
  {
    float result = 0.0f;
    for (int i = -radius; i <= radius; i ++)
    {
      int currentPos = x + i; // Posizione corrente nell'input
      if (currentPos >= 0 && currentPos < W)
      {
        result += input[currentPos] * filter[i + radius]; // Applica il filtro
      }
    }
    output[x] = result; // Salva il risultato
  }
}

```

==== Esempio di Convoluzione 2D

#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #grid(
    columns: (1fr, 1fr),
    rows: auto,
    gutter: 1em,
    //Left box
    [
      #green_heading("Descrizione")
      - *Input (I)*: matrice di 25 elementi (`I[0,0]...I[4,4]`)
      - *Filtro (F)*: matrice di 9 elementi (`F[0,0]...F[2,2]`)
      - *Output (O)*: matrice risultante dalla convoluzione di I con F.
    ]
    ,
    //Right Box
    [
      #green_heading("Formalmente")
      #math.equation(
        numbering: none,
        block: true, 
        $ O(x,y) = sum_(m=-r_x)^r_x sum_(n=-r_y)^r_y F[m,n], I[x+m,y+n] $
      )
      $r_x, r_y$: raggio del filtro 2D nelle due direzioni
    ]
    
    
  )
]

#image("images/_page_189_2.1.png")


#codly(header: [#align(center)[*CUDA Convoluzione 2D: Soluzione (non ottimale)*]])
```cpp

__global__ void cudaConvolution2D(float* input, float* output, float* filter, int W, int H, int filterSize){
  int x = blockIdx.x * blockDim.x + threadIdx.x; // Coordinata x globale del thread
  int y = blockIdx.y * blockDim.y + threadIdx.y; // Coordinata y globale del thread
  int radius = filterSize / 2; // Raggio del filtro

  if (x < W && y < H){
    float result = 0.0f;
    for (int i = -radius; i <= radius; i++){
      for (int j = -radius; j <= radius; j++){
        int currentPosX = x + j; // Posizione x corrente nell'input
        int currentPosY = y + i; // Posizione y corrente nell'input
        if (currentPosX >= 0 && currentPosX < W && currentPosY >= 0 && currentPosY < H){
          int inputIdx = currentPosY * W + currentPosX; // Indice dell'input
          int filterIdx = (i + radius) * filterSize + (j + radius); // Indice del filtro
          result += input[inputIdx] * filter[filterIdx]; // Applica il filtro
      }
    }
  }
  output[y * W + x] = result; // Salva il risultato
  }
}
```



== Riferimenti Bibliografici

#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Testi Generali")

  - Cheng, J., Grossman, M., McKercher, T. (2014).
    *Professional CUDA C Programming*. Wrox Pr Inc. ($1^\a$ edizione
  - Kirk, D. B., Hwu, W. W. (2022). 
    *Programming Massively Parallel Processors*. Morgan Kaufmann ($4^\a$ edizione)

  #green_heading("NVIDIA Docs")

  - Cuda Programming: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
  - CUDA C Best Practice Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
  - CUDA University Courses: https://developer.nvidia.com/educators/existing-courses#2
]


= Modello di Esecuzione CUDA

== Introduzione al Modello di Esecuzione CUDA

#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Modello di Esecuzione CUDA")

  In generale, un *modello di esecuzione* fornisce una *visione operativa* di come le istruzioni vengono eseguite su una specifica architettura di calcolo (nel nostro caso, le GPU).
]

// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  *Caratteristiche Principali*

  - Fornisce un'*astrazione portabile dell'architettura* (Grid, Block, Thread, Warp, SM).
  - Preserva *concetti fondamentali* tra generazioni differenti di GPU.
  - Esposizione delle *funzionalità architetturali* chiave per la programmazione CUDA.
  - Descrive come kernel, griglie e blocchi vengono effettivamente *mappati* sull'hardware GPU.
  - Basato sul *parallelismo massivo* e sul *modello SIMT* (Single Instruction, Multiple Thread).

  *Importanza*

  - Offre una *visione unificata* dell'esecuzione su diverse GPU.
  - Fornisce indicazioni utili per *l'ottimizzazione* del codice in termini di:
    - *Throughput* delle istruzioni.
    - *Accessi alla memoria.*
  - Facilita la comprensione della *relazione* tra il modello di programmazione e l'esecuzione effettiva.
  - Permette di interpretare correttamente i risultati dei profiler CUDA, collegando i fenomeni osservati (latenze, occupancy, conflitti di memoria) alla struttura del modello di esecuzione.
]



== Streaming Multiprocessor (SM)

#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Cosa sono?")
  - Gli *Streaming Multiprocessors* (SM) sono le #underline[unità fondamentali di elaborazione]
   all'interno delle GPU.
  - Ogni SM contiene diverse *unità di calcolo*, *memoria condivisa* e *altre risorse essenziali* per gestire
   l'esecuzione concorrente e parallela di migliaia di thread.
  - Il parallelismo hardware delle GPU è ottenuto attraverso la *replica* di questo blocco architetturale.
]


#figure(image("images/_page_3_Figure_5_2.2.jpeg"))

#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("1. CUDA Cores")
    - Unità di elaborazione che eseguono istruzioni aritmetico/logiche.

  #green_heading("2. Shared Memory/L1 Cache")
  - Memoria ad alta velocità condivisa tra i thread di un blocco.

  #green_heading("3. Register Files")
  - Memoria privata di ogni thread per dati temporanei.

  #green_heading("4. Load/Store Units (LD/ST)")
  - Gestiscono il trasferimento dati da/verso la memoria.

  #green_heading("5. Special Function Units (SFU)")
  - Accelerano calcoli matematici complessi (funzioni trascendenti).

  #green_heading("6. Warp Scheduler")
  - Seleziona thread pronti per l'esecuzione nell'SM.

  #green_heading("7. Dispatch Unit")
  - Assegna i thread selezionati alle unità di esecuzione.

  #green_heading("8. Instruction Cache")
  - Memorizza temporaneamente le istruzioni usate di frequente.


]


#figure(image("images/_page_4_Figure_18_2.2.png", width: 45%))


== CUDA Core - Unità di Elaborazione CUDA

#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Cos'è un CUDA Core?")
  - Un *CUDA Core* è l'*unità di elaborazione* di base all'interno di un SM di una GPU NVIDIA.
  - L'architettura e la funzionalità dei CUDA Core sono evolute nel tempo, 
    passando da unità generiche a unità specializzate.
  #figure(image("images/_page_5_Picture_4_2.2.jpeg", width: 30%))

  #green_heading("Composizione e Funzionamento (Architettura Fermi e Precedenti)")

  - Inizialmente, i CUDA Core erano unità di calcolo relativamente semplici, in grado
    di eseguire sia operazioni intere (INT) che in virgola mobile (FP) in un ciclo di 
    clock (fully pipelined, non simultaneamente).
    - *ALU (Arithmetic Logic Unit):* Ogni CUDA Core contiene un'unità logico-aritmetica 
    che esegue operazioni matematiche di base come addizioni, sottrazioni, moltiplicazion
    e operazioni logiche.
    - *FPU (Floating Point Unit)*: Include anche una FPU per gestire le operazioni 
    in virgola mobile, supportando principalmente calcoli a precisione singola (FP32).
  - I CUDA Core usano *registri condivisi* a livello di Streaming Multiprocessor
    per memorizzare temporaneamente dati durante l'esecuzione dei thread.
]

// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Evoluzione dell'Architettura (Kepler e successive)")
  // Elenco puntato principale
    #list(marker: [•], body-indent: 0.5em)[
      Dall'architettura Kepler, NVIDIA ha introdotto la *specializzazione delle unità di calcolo* all'interno di uno SM:
    ]
    
    #v(0.5em)

    // --- Griglia per la sezione strutturata (General, AI, Grafica) ---
    // La griglia ha 2 colonne: etichette ruotate e contenuto
    #grid(
      columns: (auto, 1fr),
      column-gutter: 1em,
      row-gutter: 0.8em,
      
      // --- Riga 1: Separatore tratteggiato ---
      grid.cell(colspan: 2, line(length: 100%, stroke: (dash: "dashed", thickness: 0.5pt, paint: gray))),

      // --- Riga 2: General ---
      grid.cell(align: center + horizon)[
        #rotate(-90deg, reflow: true)[*General*]
      ],
      [
        #set list(marker: circle(radius: 1.5pt, stroke: black))
        #list(
          [*Unità FP64*: Dedicate alle operazioni in virgola mobile a _doppia precisione_.],
          [*Unità FP32*: Dedicate alle operazioni in virgola mobile a _singola precisione_.],
          [*Unità INT*: Dedicate alle _operazioni intere_.]
        )
      ],

      // --- Riga 3: Separatore tratteggiato ---
      grid.cell(colspan: 2, line(length: 100%, stroke: (dash: "dashed", thickness: 0.5pt, paint: gray))),

      // --- Riga 4: AI ---
      grid.cell(align: center + horizon)[
        #rotate(-90deg, reflow: true)[*AI*]
      ],
      [
        #set list(marker: circle(radius: 1.5pt, stroke: black))
        #list(
          [*Tensor Core - TC* (Architettura Volta e successive): Unità specializzate particolarmente ottimizzate per moltiplicazioni fra matrici in _precisione ridotta/mista_ (FP32, FP16, TF32, INT8, etc.).]
        )
      ],

      // --- Riga 5: Separatore tratteggiato ---
      grid.cell(colspan: 2, line(length: 100%, stroke: (dash: "dashed", thickness: 0.5pt, paint: gray))),

      // --- Riga 6: Grafica ---
      grid.cell(align: center + horizon)[
        #rotate(-90deg, reflow: true)[*Grafica*]
      ],
      [
        #set list(marker: circle(radius: 1.5pt, stroke: black))
        #list(
          [*Ray Tracing Core - RT* (Ampere e successive): Unità dedicate per l'accelerazione del _ray tracing_.],
          [*Unità di Texture*: Ottimizzate per gestire _texture_ e _operazioni di filtraggio_.],
          [*Unità di Rasterizzazione*: Utilizzate per la _rasterizzazione_ delle immagini durante il rendering.]
        )
      ]
    )

  #green_heading("Ruolo del Modello CUDA")
  - *Esecuzione Parallela*: Ogni unità di elaborazione esegue un thread in 
    #underline[parallelo] con altri nel medesimo SM.

]

// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Differenze rispetto alle CPU")
    - *Semplicità Architetturale*: Le varie unità di gestione all'interno di un SM 
      sono più semplici rispetto ai core delle CPU, #underline[senza unità di controllo 
      complesse], permettendo una maggiore densità di unità specializzate.
  - *Specializzazione*: Mentre le CPU sono general purpose, le GPU, attraverso 
    i CUDA Core e le unità specializzate, offrono performance elevate anche 
    per compiti specifici come l'*Intelligenza Artificale* ed il *rendering grafico*.
]

== Streaming Multiprocessor (SM) - Evoluzione

#figure(image("images/Evoluzione_1.png"))
#figure(image("images/Evoluzione_2.png"))
#figure(image("images/Evoluzione_3.png"))

// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  - *Aumento di SM e CUDA Core*: Ogni generazione ha generalmente aumentato
    il numero di SM e CUDA Core.
  - *Miglioramento del Parallelismo*: L'aumento delle unità di elaborazione 
    permettono un #underline[maggiore parallelismo], migliorando le prestazioni 
    complessive della GPU.
  - *Calcolo CUDA Core Totali*: Totale CUDA Core = (SM per GPU) x (CUDA Core per SM)
]

#block(
  width: 100%,
  fill: rgb("#F3F3F3"),
  radius: 8pt,
  inset: 15pt,
)[
  // Funzione helper per formattare la prima colonna (Nome Architettura + Serie)
  #let arch-cell(name, series) = {
    text(font: "Liberation Mono", weight: "bold", fill: nvidia-green, size: 1.1em)[#name]
    h(0.5em)
    text(weight: "bold")[#series]
  }

  // Funzione helper per i numeri (font monospaziato)
  #let num-cell(val) = {
    text(font: "Liberation Mono", size: 1em)[#val]
  }

  #table(
    columns: (1fr, auto, auto, auto),
    column-gutter: 1em,
    row-gutter: -0.5em,
    stroke: none,
    align: (x, y) => if x == 0 { left + horizon } else { center + horizon },

    // --- Intestazioni ---
    [*Architettura*],
    [*SM per GPU*],
    align(center)[*CUDA Cores*\ *FP32 per SM*],
    align(center)[*Totale CUDA*\ *FP32 Cores*],

    // --- Righe Dati ---
    arch-cell("Tesla", "(GTX 200 series)"),       num-cell("30"),  num-cell("8"),   num-cell("240"),
    arch-cell("Fermi", "(GTX 400/500 series)"),   num-cell("16"),  num-cell("32"),  num-cell("512"),
    arch-cell("Kepler", "(GTX 600/700 series)"),  num-cell("15"),  num-cell("192"), num-cell("2880"),
    arch-cell("Maxwell", "(GTX 900 series)"),     num-cell("16"),  num-cell("128"), num-cell("2048"),
    arch-cell("Pascal", "(GTX 10 series)"),       num-cell("20"),  num-cell("128"), num-cell("2560"),
    arch-cell("Volta", "(Tesla V100)"),           num-cell("80"),  num-cell("64"),  num-cell("5120"),
    arch-cell("Turing", "(RTX 20 series)"),       num-cell("72"),  num-cell("64"),  num-cell("4608"),
    arch-cell("Ampere", "(RTX 30 series)"),       num-cell("84"),  num-cell("128"), num-cell("10752"),
    arch-cell("Ada Lovelace", "(RTX 40 series)"), num-cell("128"), num-cell("128"), num-cell("16384"),
    arch-cell("Hopper", "(GH series)"),           num-cell("144"), num-cell("128"), num-cell("18432"),
    arch-cell("Blackwell", "(RTX 50 series)"),    num-cell("170"), num-cell("128"), num-cell("21760"),
  )

  #v(1em)
  
  // Nota a piè di pagina
  #text(size: 0.95em)[
    *Nota:* I valori mostrati sono tipici dei modelli di punta. Possono esserci variazioni tra i diversi modelli di una stessa serie.
  ]
]


== Tensor Core: Acceleratori per l'Intelligenza Artificiale (Volta+)

// Griglia
#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  // Box sinistro
  {
    block(
        stroke: 1pt + light_green,
        radius: 0.8em,
        inset: 1.5em,
        width: 100%,
        breakable: false,
        // height: 22em
      )[
        #text(fill: light_green, weight: "bold")[Cosa sono i Tensor Core?]
        
        - *Unità di elaborazione specializzata* per operazioni tensoriali (array multidimensionali).
        - Progettata per accelerare calcoli di *AI* e *HPC*  (Riduzione dei tempi di training e inferenza).
        - Presenti in GPU NVIDIA RTX da Volta (2017) in poi.

        #green_heading("Caratteristiche")
        - Esegue operazioni *matrice-matrice* (es. GEMM General Matrix Multiply) 
          in *precisione mista*.
        - Supporta formati *FP8*, *FP16*, *FP32*, *FP64*, *INT8*, *INT4*, 
          *BF16* e nuovi formati come *TF32 (TensorFloat-32).*
        - Offrono un significativo *speedup* nel calcolo senza compromettere l'accuratezza.
      ]
      // GreenBlock
      block(
        fill: rgb("#F3F3F3"),
        radius: 0.8em,
        inset: 1.5em,
        width: 100%,
        // height: 21em,
        breakable: false
      )[
        #green_heading("Evoluzione")
      - *Miglioramenti*: Volta → ..→ .. → Hopper → Blackwell
      - Integrazione con CUDA, cuDNN, TensorRT
      ]
  },
  //Box destro
  figure(image("images/_page_12_Picture_12_2.2.jpeg"))
)

// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  - *Fused Multiply-Add (FMA):* Un'operazione che combina una moltiplicazione 
    e un'addizione di *scalari* in un unico passo, eseguendo . Un CUDA core 
    esegue 1 FMA per ciclo di clock in FP32.
  - *Matrix Multiply-Accumulate (MMA):* Operazione che calcola il prodotto 
    di due *matrici* e somma il risultato a una terza matrice, eseguendo
  - Per matrici ( ), ( ) e ( ), l'operazione produce ( ) e richiede operazioni 
    FMA, dove ogni elemento di necessita di moltiplicazioni-addizioni.
]


#figure(image("images/_page_13_Figure_4_2.2.jpeg"))

// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Esecuzione Parallela")
  - Ogni Tensor Core esegue *64 operazioni FMA (4x4x4)* in un singolo ciclo di clock, grazie al parallelismo interno.
- Per operazioni su matrici più grandi, queste vengono *decomposte in sottomatrici 4x4*.
- Più operazioni 4x4 vengono eseguite *in parallelo su diversi Tensor Cores*.
]



== Evoluzione dei NVIDIA Tensor Core

Le generazioni più recenti di GPU hanno ampliato la flessibilità dei Tensor Cores, supportando *dimensioni di matrici più grandi e/o sparse* con un maggiore numero di formati numerici.

#figure(image("images/_page_14_Figure_2_2.2.jpeg"))

// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Impatto delle Approssimazioni")
  - *Accelerazione* significativa dei calcoli
  - *Riduzione* del consumo di memoria
  - *Perdita di Precisione*: di è dimostrato che ha un impatto minimo
    sull'accuratezza finale dei modelli di deep learning
]

== Organizzazione e gestione dei thread

=== SM, Thread Blocks e Risorse

// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Parallelismo Hardware")
  - Più SM per GPU permettono l'*esecuzione simultanea* di 
    migliaia di thread (anche da kernel differenti).

  #green_heading("Distribuzione dei Thread Blocks")
  - Quando un kernel viene lanciato, i blocchi di vengono *automaticamente
    e dinamicamente distribuiti dal GigaThread Engine* (scheduler globale) 
    agli SM disponibili.
  - Le variabili di identificazione e dimensione 
    (*gridDim*, *blockIdx*, *blockDim*, e *threadIdx)* sono rese disponibili 
    ad ogni thread e condivise nello stesso SM.
  - Una volta assegnato a un SM, un blocco rimane vincolato 
    a quell'SM *per tutta la durata dell'esecuzione*.
  
  #green_heading("Gestione delle Risorse e Scheduling")
  - *Più blocchi di thread* possono essere assegnati 
    *allo stesso SM* contemporaneamente.
  - Lo scheduling dei blocchi dipende dalla *disponibilità 
    delle risorse* dell'SM (registri, memoria condivisa) e 
    dai *limiti architetturali* di ciascun SM (max blocks, max threads, etc.).
  - Tipicamente, la maggior parte delle grid contiene 
    *molti più blocchi di quanti possano essere eseguiti* in parallelo 
    sugli SM disponibili.
  - Il *runtime system* mantiene quindi una coda di blocchi 
    in attesa, assegnandone di nuovi agli SM non appena quelli 
    precedenti terminano l'esecuzione.
]


=== Corrispondenza tra Vista Logica e Vista Hardware

#figure(image("images/_page_17_Picture_1_2.2.jpeg"))


#figure(image("images/_page_18_Picture_1_2.2.jpeg"))


#figure(image("images/_page_19_Figure_1_2.2.jpeg"))

=== Distribuzione dei Blocchi su Streaming Multiprocessors

#figure(image("images/_page_20_Picture_1_2.2.jpeg"))

Supponiamo di dover realizzare un algoritmo parallelo che effettui il calcolo parallelo su un'immagine.

// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  - Il Gigathread Engine *smista* i blocchi di thread agli SM in base alle risorse disponibili.
  - CUDA non garantisce l'ordine di esecuzione e non è possibile scambiare dati tra i blocchi.
  - Ogni blocco viene elaborato in modo *indipendente*.
]
#figure(image("images/_page_21_Picture_1_2.2.png"))
Quando un blocco completa l'esecuzione e libera le risorse, un nuovo blocco viene schedulato al suo posto nell'SM, e questo processo continua fino a quando tutti i blocchi del grid non sono stati elaborati.
#figure(image("images/_page_22_Picture_2_2.2.jpeg"))

=== Concetto di Wave in CUDA

// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Cosa si intende per Wave?")
  - Un "*Wave*" rappresenta l'insieme dei blocchi di thread che vengono
    eseguiti #underline[simultaneamente] su tutti gli SM della GPU in un dato momento.
  - La *Full Wave Capacity*, invece, rappresenta la *#underline[capacità teorica 
    massima]* della GPU, ossia il numero totale di blocchi che possono
    essere residenti simultaneamente su tutti gli SM.
    ```
    Full Wave Capacity = (Numero di SM) * (Numero massimo di blocchi attivi per SM)
    ```

    ⚠️ *Attenzione*: Questo numero massimo di blocchi *#underline[dipende 
    dall'architettura GPU] (Compute Capability)* e *#underline[dalle risorse 
    richieste da ciascun blocco]* (come registri, memoria condivisa), 
    che influenzano *l'occupancy* (lo vedremo).
]

// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Full Wave vs Partial Wave")
  - *Full Wave*: tutti gli SM sono occupati al massimo della loro capacità → utilizzo 100%
  - *Partial Wave*: solo parte degli SM è occupata, oppure non tutti al massimo → utilizzo < 100%
  - *Esempio*: GPU con 80 SM e fino a 4 blocchi attivi per SM → *Full Wave Capacity* = 80 x 4 = 320 blocchi simultanei.
    - Se il kernel lancia:
      - *320 blocchi* → esegue in *1 full wave* (320/320 = 100% utilizzo)
      - *100 blocchi* → esegue in *1 partial wave* (100/320 = ~31% utilizzo)
      - Cosa succede se il numero di blocchi è *superiore alla capacità massima*? (es. 500 blocchi)
]

==== Numero di Waves per un Kernel CUDA

// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Calcolo del Numero di Waves")
  - Quando si lanciano più blocchi di quelli che la GPU può gestire 
    simultaneamente, l'esecuzione avviene in più *ondate successive (waves)*:
  - Il numero di blocchi attivi è una proprietà *statica*, 
    determinata dall'architettura e dalle risorse richieste dal kernel.
    ```
  Numero di waves = ⌈(Blocchi totali) / (Full wave capacity)⌉
  ```
]
// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  *Esempio*

  Consideriamo una GPU con *8 SM* e un kernel con *12 blocchi totali*
  che consente *1 solo blocco attivo per SM*.

  - *Full wave capacity =* 8 SM x 1 blocco/SM = 8 blocchi
  - *Numero di waves* = ⌈12 / 8⌉ = 2 waves

  *Esecuzione*:

  - *Wave 1 (full wave):* 8 blocchi su 8 SM → *utilizzo 100%*
  - *Wave 2 (partial wave):* 4 blocchi su 8 SM *→ utilizzo 50%*
  - *Efficienza media di esecuzione:* (8 + 4) / (8 + 8) = 12/16 = 75%
  - Il secondo wave è un esempio di *tail effect*.
] 
#figure(image("images/_page_24_Figure_14_2.2.jpeg", width: 66%))

=== Scalabilità in CUDA

// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Cosa si intende?")
  - Per *scalabilità* in CUDA ci si riferisce alla capacità di un'applicazione
    di migliorare le prestazioni proporzionalmente all'*aumento delle risorse*
    hardware disponibili.
  - *Più SM* disponibili = *Più blocchi eseguiti* contemporaneamente
    \= *Maggiore Parallelismo*.
  - *Nessuna modifica al codice* richiesta per sfruttare hardware più potente.
]

#figure(image("images/_page_25_Figure_5_2.2.jpeg"))

== Modello di Esecuzione SIMT e Warp

=== Modello di Esecuzione: SIMD
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("SIMD (Single Instruction, Multiple Data)")
  - È un *modello di esecuzione parallela* in cui una singola istruzione opera
    simultaneamente su più elementi di dato, utilizzando *unità vettoriali
    dedicate* (vector units) presenti nei core della CPU
- Utilizza *registri vettoriali* che possono contenere più elementi
  (es. 4 float, 8 int16, 16 byte).
- Il programma segue un *flusso di controllo centralizzato* 
  (singolo thread di controllo).
- *Limitazioni*:
  - Larghezza vettoriale *fissa* nell'hardware (es. AVX-512 
    consente 512 bit), limitando gli elementi per istruzione.
  - Tutti gli elementi vettoriali in un vettore vengono elaborati
    in *lockstep* (perfettamente sincroni).
  - *La divergenza non è ammessa:* se occorrono percorsi 
    condizionali (*if*-*else*), si impiegano *maschere vettoriali*
    che selezionano gli elementi su cui applicare l'operazione.

]

#codly(header: [#align(center)[*Somma di Due Array (SIMD con Neon intrinsics - ARM)*]])

```
void array_sum(uint32_t *a, uint32_t *b, uint32_t *c, int n){ 
  for(int i=0; i<n; i+=4) {
    //calcola c[i], c[i+1], c[i+2], c[i+3] 
    uint32x4_t a4 = vld1q_u32(a+i); 
    uint32x4_t b4 = vld1q_u32(b+i); 
    uint32x4_t c4 = vaddq_u32(a4,b4);
    vst1q_u32(c+i,c4);
  }}
```
*N.B.* I dati vengono suddivisi in vettori di dimensione fissa e 
il loop elabora questi vettori utilizzando istruzioni _intrinsics_ 
con nomenclatura specifica dell'architettura. 

=== Modello di Esecuzione: SIMT

// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("SIMT (Single Instruction, Multiple Thread)")
  - *Modello ibrido* adottato in CUDA che combina parallelismo 
    multi-thread con esecuzione SIMD-like.
  - *Caratteristiche Chiave*:
    - A differenza del SIMD, non ha un controllo centralizzato delle istruzioni.
    - Ogni thread possiede un proprio *Program Counter (PC)*, 
      *registri* e *stato* indipendenti (maggiore flessibilità).
    - *Supporta divergenza* del flusso di controllo (thread possono 
      avere percorsi di esecuzione indipendenti).
    - Hardware gestisce *automaticamente* la divergenza (trasparente al programmatore).
  - *Implementazione*
    - In CUDA, i thread sono organizzati in #underline[gruppi di 32] 
      chiamati #circled[*warps*] (unità minima di esecuzione in un SM).
    - I thread in un warp iniziano insieme allo *stesso 
      indirizzo del programma (PC),* ma #underline[possono divergere].
    - Divergenza in un warp causa *esecuzione seriale dei percorsi 
      diversi*, riducendo l'efficienza (da evitare).
    - La divergenza è *gestita automaticamente dall'hardware*, ma 
      con un impatto negativo sulle prestazioni.
]

#codly(header: [#align(center)[*Somma di Due Array (SIMT)*]])
```cpp
__global__ void array_sum(float *A, float *B, float *C, int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) C[idx] = A[idx] + B[idx]; //Questa riga rappresenta l'essenza del SIMT
} 
```
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #align(center)[*Perché 32 Thread in un Warp CUDA?*]
  - *Efficienza Hardware*: Massimizza l'utilizzo delle risorse hardware dell'SM
    - Warp troppo piccolo sarebbe inefficiente, mentre uno troppo grande
      complicherebbe lo scheduling e potrebbe sovraccaricare gli SM / la memoria.
  - *Efficienza della Memoria*: Un warp di 32 thread accede a indirizzi di memoria
    consecutivi, permettendo aggregazioni in poche transazioni e massimizzando
    l'efficienza delle linee di connessione per evitare accessi parziali.
  - *Flessibilità Software*: Offre una granularità gestibile per il controllo
    della divergenza e per il bilanciamento del carico di 
    lavoro tra thread.
  - *Adattabilità*: Questa dimensione si è dimostrata efficace per varie
    generazioni di GPU NVIDIA, pur rimanendo aperta a future evoluzioni.
]


#block(
  width: 100%,
  fill: my_gray,
  radius: 8pt,
  inset: 15pt,
  breakable: false,
)[
  #align(center)[#text(size: 1.3em)[*Modello di Esecuzione: SIMD vs. SIMT*]]
  #grid(
    columns: (auto, 1fr, 1fr),
    column-gutter: 1.5em,
    row-gutter: 0.8em,
    
    // --- Intestazione ---
    [], // Cella vuota sopra la prima colonna
    align(center, text(fill: device-blue, weight: "bold", size: 1.1em)[SIMD]),
    align(center, text(fill: nvidia-green, weight: "bold", size: 1.1em)[SIMT]),

    // --- Riga 1: Unità di Esecuzione ---
    [*Unità di Esecuzione*],
    [Un *singolo thread* controlla vettori di dimensione fissa],
    [*Molti thread* leggeri raggruppati in warp (32 thread)],

    dashed-line,

    // --- Riga 2: Registri ---
    [*Registri*],
    [*Registri vettoriali* condivisi tra le unità di calcolo],
    [Set completo di *registri per thread*],

    dashed-line,

    // --- Riga 3: Flessibilità ---
    [*Flessibilità*],
    [*Bassa*: Stessa operazione per tutti gli elementi vettore],
    [*Alta*: Ogni thread può eseguire operazioni e percorsi indipendenti],

    dashed-line,

    // --- Riga 4: Indipendenza ---
    [*Indipendenza*],
    [*Non applicabile*, controllo centralizzato],
    [Ogni thread mantiene il *proprio stato di esecuzione* (vedi nota\*)],

    dashed-line,

    // --- Riga 5: Branching ---
    [*Branching*],
    [Gestito *esplicitamente* con *maschere* (no divergenza)],
    [Gestito *via hardware* con *thread masking automatico*],

    dashed-line,

    // --- Riga 6: Scalabilità ---
    [*Scalabilità*],
    [*Limitata* dalla larghezza vettoriale],
    [*Massiva* (migliaia/milioni di thread)],

    dashed-line,

    // --- Riga 7: Sincronizzazione ---
    [*Sincronizzazione*],
    [*Intrinseca* (lock-step automatico)],
    [*Esplicita* (es., #text(font: "Liberation Mono",
      fill: nvidia-green, weight: "bold")[\_\_syncthreads])],

    dashed-line,

    // --- Riga 8: Utilizzo Tipico ---
    [*Utilizzo Tipico*],
    [*Estensioni CPU* (SSE, AVX, NEON)],
    [*GPU* Computing (CUDA, OpenCL)],
  )
]

*Nota*: Con l'architettura Volta, NVIDIA ha introdotto l'Independent Thread Scheduling.
- *Pre-Volta*, un Program Counter (PC) era condiviso per warp; 
- *Post-Volta*, ogni thread ha il proprio PC, migliorando la gestione della divergenza e la flessibilità d'esecuzione.

=== Modello di Esecuzione Gerarchico di CUDA

// --- Riquadro Esterno Tratteggiato ---
#block(
  width: 100%,
  stroke: (dash: "dashed", thickness: 1pt, paint: black),
  radius: 8pt,
  inset: 15pt,
)[
  // Titolo
  #text(weight: "bold")[Livello di Programmazione]
  
  #v(0.8em)

  // Griglia a due colonne per i codici
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 1em,

    ```cpp
    __global__ void array_sum(float *A, float *B, float *C, int N) {
      int idx = blockDim.x * blockIdx.x + threadIdx.x;
      if (idx < N) C[idx] = A[idx] + B[idx];
    }
    ```,
    ```cpp
    int main(int argc, char **argv){
      // ...
      // Chiamata del kernel
      array_sum<<<gridDim ,blockDim >>>(args);
    }
    ```
  )
]

#block(
  width: 100%,
  stroke: (dash: "dashed", thickness: 1pt, paint: black),
  radius: 8pt,
  inset: 15pt,
)[
  #figure(image("images/_page_31_Figure_2_2.2.jpeg"))
]
#figure(image("images/_page_32_Picture_1_2.2.jpeg"))

=== Warp: L'Unità Fondamentale di Esecuzione nelle SM

// Griglia 
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    // Box sinistro
    block(
        stroke: 1pt + nvidia-green,
        radius: 0.8em,
        inset: 1.5em,
        width: 100%,
        breakable: false,
        // height: 22em
      )[
        #set text(size: 8pt)
        #green_heading("Distribuzione dei Thread Block")
        - Quando si lancia una griglia di thread block, questi 
          vengono *distribuiti* tra i diversi SM disponibili.
        #green_heading("Partizionamento in Warp")
        I thread di un thread block vengono suddivisi in 
        *warp di 32 thread (con ID consecutivi).*

        #green_heading("Esecuzione SIMT")

        - I thread in un warp eseguono la *stessa istruzione* 
          su *dati diversi*, con possibilità di *divergenza*.

        #green_heading("Esecuzione Logica vs Fisica")

        - Thread eseguiti in parallelo *logicamente*, ma non sempre fisicamente.

        #green_heading("Scheduling Dinamico (Warp Scheduler)")

        - L'SM gestisce *dinamicamente* l'esecuzione di un numero 
          limitato di warp, switchando efficientemente tra di essi.

        #green_heading("Sincronizzazione")

        - Possibile all'interno di un thread block, ma non tra thread block diversi.

      ],
    //Box destro
    image("images/_page_33_Figure_13_2.2.jpeg")
  )

==== Organizzazione dei Thread e Warp


// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Thread Blocks e Warp")
  - *Punto di Vista Logico:* Un blocco di thread è una collezione 
    di thread organizzati in un layout 1D, 2D o 3D.
  - *Punto di Vista Hardware:* Un blocco di thread è una #underline[collezione 
    1D di warp]. I thread in un blocco sono organizzati in un layout 
    1D e ogni insieme di 32 thread consecutivi (con ID consecutivi) forma un warp.
]


// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  *Esempio 1D*

  Un blocco 1D con 128 thread viene suddiviso in 4 warp, ognuno composto da 32 thread (*ID Consecutivi*).

  ```
  Warp 0: thread 0, thread 1, thread 2, ... thread 31
  Warp 1: thread 32, thread 33, thread 34, ... thread 63
  Warp 2: thread 64, thread 65, thread 66, ... thread 95
  Warp 3: thread 96, thread 97, thread 98, ... thread 127
  ```
]
#align(center)[*Thread Block N (Caso 1D)*]
#figure(
  image("images/_page_34_Figure_8_2.2.jpeg")
  )

// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  *Mapping Multidimensionale (2D e 3D)*
  - Il *programmatore* usa *`threadIdx`* e *`blockDim`* per identificare 
    i thread nel #underline[layout logico].
  - Il *runtime CUDA* si occupa automaticamente di linearizzare 
    gli indici multidimensionali in ordine row-major, raggruppare 
    i thread in warp, gestire il mapping hardware.
  - L'ID di un thread in un blocco multidimensionale viene 
    calcolato usando *threadIdx* e *blockDim*

    - *Caso 2D*: #h(1em) `threadIdx.y * blockDim.x + threadIdx.x`
    - *Caso 3D*: #h(1em) `threadIdx.z * blockDim.y * blockDim.x +
          + threadIdx.y * blockDim.x + threadIdx.x`
    - *Calcolo del Numero di Warp*: #h(1em) `ceil(ThreadsPerBlock/warpSize)`
  - L'hardware alloca #underline[sempre] un numero *discreto* di warp.
]

// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  *Mapping Multidimensionale (Caso 2D)*

  - Esempio 2D: Un thread block 2D con 40 thread in x e 2 in y 
    (80 thread totali) richiederà 3 warp (96 thread hardware). 
    L'ultimo semi-warp (16 thread) sarà inattivo, consumando comunque risorse.
]



#figure(image("images/_page_36_Figure_6_2.2.jpeg"))

// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Warp: L'Unità Fondamentale di Esecuzione nell'SM")

  - Un warp viene *assegnato* a una sub-partition, solitamente in base al 
    suo ID, dove rimane #underline[fino al completamento].
  - Una sub-partition gestisce un *"pool" di warp concorrenti* di #underline[dimensione 
    fissa] (es., Turing 8 warp, Volta 16 warp).
]



#figure(image("images/_page_37_Figure_3_2.2.jpeg"))

==== Compute Capability (CC) - Limiti su Blocchi e Thread
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  - La *Compute Capability (CC)* di NVIDIA è un numero che identifica 
    le *caratteristiche* e le *capacità* di una GPU NVIDIA in termini 
    di funzionalità supportate e limiti hardware.
  - È composta da *due numeri*: il numero principale indica la *generazione*
    dell'architettura, mentre il numero secondario indica *revisioni* e 
    *miglioramenti* all'interno di quella generazione.
]

// --- Helper Functions ---
// Per i nomi delle architetture (Verde, Bold, Mono)
#let arch(name) = text(font: "Liberation Mono", weight: "bold", fill: nvidia-green, name)
// Per i dati numerici (Nero, Mono)
#let num(val) = text(font: "Liberation Mono", val)
// Per le intestazioni (Bold, Sans-serif)
#let hdr(txt) = text(weight: "bold", txt)

#block(
  width: 100%,
  fill: my_gray,
  radius: 8pt,
  inset: 15pt,
  breakable: false,
)[
  #table(
    columns: (auto, auto, auto, auto, auto, auto),
    // Allinea tutto al centro orizzontalmente e verticalmente
    align: center + horizon, 
    stroke: none,
    column-gutter: 0.1em,
    row-gutter: -0.5em, // Spaziatura verticale comoda come nell'immagine

    // --- Intestazioni ---
    // Uso le interruzioni di riga manuali (\) per replicare l'impaginazione
    hdr[Compute\ Capability],
    hdr[Architettura],
    hdr[Warp Size],
    hdr[Max Blocchi\ per SM\*],
    hdr[Max Warp\ per SM\*],
    hdr[Max Threads\ per SM\*],

    // --- Riga 1 ---
    num[1.x], arch[Tesla], num[32], num[8], num[24/32], num[768/1024],
    
    // --- Riga 2 ---
    num[2.x], arch[Fermi], num[32], num[8], num[48], num[1536],

    // --- Riga 3 ---
    num[3.x], arch[Kepler], num[32], num[16], num[64], num[2048],

    // --- Riga 4 ---
    num[5.x], arch[Maxwell], num[32], num[32], num[64], num[2048],

    // --- Riga 5 ---
    num[6.x], arch[Pascal], num[32], num[32], num[64], num[2048],

    // --- Riga 6 ---
    num[7.x], arch[Volta/Turing], num[32], num[16/32], num[32/64], num[1024/2048],

    // --- Riga 7 ---
    num[8.x], arch[Ampere/Ada], num[32], num[16/24], num[48/64], num[1536/2048],

    // --- Riga 8 ---
    num[9.x], arch[Hopper], num[32], num[32], num[64], num[2048],

    // --- Riga 9 ---
    num[10.x/12.x], arch[Blackwell], num[32], num[32], num[64/48], num[2048/1536],
  )
]

#align(center)[https://en.wikipedia.org/wiki/CUDA=Version_features_and_specifications]


==== Warp: Contesto di Esecuzione
#let c-red = rgb("#ff0000")

// --- Funzioni Helper per i Box ---

// Box Esterno (Linea continua verde)
#let main-box(body) = block(
  stroke: (paint: nvidia-green, thickness: 1.5pt),
  radius: 8pt,
  inset: 15pt,
  width: 100%,
  body
)

// Box Interni (Linea tratteggiata)
#let dashed-box(color, body) = block(
  stroke: (paint: color, dash: "dashed", thickness: 1pt),
  radius: 5pt,
  inset: 8pt,
  width: 100%,
  body
)

// --- Layout del Documento ---


// Griglia
#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  // Box sinistro
  main-box[
  #set text(size: 8pt)
  // Punto elenco principale
  #list(marker: [•], body-indent: 0.5em)[
    Il contesto di *esecuzione locale* di un warp in un SM contiene:
  ]

  #v(0.5em)

  // --- PRIMO BLOCCO: VERDE (con testo ruotato) ---
  #dashed-box(nvidia-green)[
    #grid(
      columns: (auto, 1fr),
      column-gutter: 0.8em,
      
      // Colonna 1: Etichetta Ruotata
      align(center + horizon)[
        #rotate(-90deg, reflow: true)[
          #text(fill: nvidia-green, size: 8pt)[Volta- : Per warp]\
          #text(fill: nvidia-green, size: 8pt)[Volta+ : Per thread]
        ]
      ],
      
      // Colonna 2: Lista
      [
        #set list(marker: circle(radius: 1.5pt, stroke: black))
        #list(
          spacing: 0.6em,
          [*Program Counter (PC)*: Indica l’indirizzo della prossima istruzione da eseguire.],
          [*Call Stack*: Struttura dati che memorizza le informazioni sulle chiamate di funzione, inclusi gli indirizzi di ritorno, gli argomenti, array e strutture dati più grandi.]
        )
      ]
    )
  ]

  #v(0.3em)

  // --- SECONDO BLOCCO: ROSSO (Registri) ---
  #dashed-box(c-red)[
    #set list(marker: circle(radius: 1.5pt, stroke: black))
    // Aggiungo un h(2.5em) manuale o uso un grid simile sopra per allineare perfettamente i pallini
    // se necessario, ma qui lascio il flow naturale del blocco.
    #list(
      [*Registri*: Memoria veloce e #underline[privata] per ogni thread, utilizzata per memorizzare variabili e dati temporanei.]
    )
  ]

  #v(0.2em) // Spazio ridotto tra i due blocchi rossi

  // --- TERZO BLOCCO: ROSSO (Memoria Condivisa) ---
  #dashed-box(c-red)[
    #set list(marker: circle(radius: 1.5pt, stroke: black))
    #list(
      [*Memoria Condivisa*: Memoria veloce e #underline[condivisa] tra i thread di un blocco utile per comunicare.]
    )
  ]

  #v(0.3em)

  // --- LISTA RIMANENTE (Senza box) ---
  #set list(marker: circle(radius: 1.5pt, stroke: black))
  #list(
    spacing: 0.6em,
    [*Thread Mask*: Indica quali thread del warp sono attivi o inattivi durante l’esecuzione di un’istruzione.],
    [*Stato di Esecuzione*: Informazioni sullo stato corrente del warp (es. in esecuzione/in stallo/eleggibile).],
    [*Warp ID*: Identificatore che consente di distinguere i warp e calcolare l’offset nel register file per ogni thread nel warp.]
  )

  #v(0.8em)

  // --- FOOTER ---
  #list(marker: [•], body-indent: 0.5em)[
    L'SM mantiene *on-chip* il contesto di ogni warp #underline[per tutta la sua durata], quindi il *cambio di contesto è senza costo*.
  ]
],
  //Box destro
  image("images/_page_39_Figure_12_2.2.png")
)






==== Parallelismo a Livello di Warp nell'SM

#figure(image("images/_page_41_2.2.png"))

=== Classificazione dei Thread Block e Warp

// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Thread Block Attivo (Active Block)")

  - Un thread block viene considerato *attivo* (o *residente*) quando gli vengono 
    allocate risorse di calcolo di un SM come registri e memoria condivisa 
    (non significa che tutti i suoi warp siano in esecuzione simultaneamente 
    sulle unità).
  - I warp contenuti in un thread block attivo sono chiamati *warp attivi.*
  - Il numero di blocchi/warp attivi in ciascun istante è *limitato dalle 
    risorse* dell'SM (compute capability).
]


// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Tipi di Warp Attivi")

  + *Warp Selezionato (Selected Warp)*
    - Un warp #underline[in esecuzione] attiva su un'unità di 
      elaborazione (FP32, INT32, Tensor Core, etc.).
  + *Warp in Stallo (Stalled Warp)*
    - Un warp #underline[in attesa] di dati o risorse, impossibilitato 
      a proseguire l'esecuzione.
    - Cause comuni: latenza di memoria, dipendenze da istruzioni, sincronizzazioni.
  + *Warp Eleggibile/Candidato (Eligible Warp)*
    - Un warp #underline[pronto] (ma ancora non scelto) per l'esecuzione, 
      con tutte le risorse necessarie disponibili.
    - Condizioni per l'eleggibilità:
      - *Disponibilità Risorse*: I thread del warp devono essere allocabili 
        sulle unità di esecuzione disponibili.
      - *Prontezza Dati*: Gli argomenti dell'istruzione corrente devono 
        essere pronti (es. dati dalla memoria).
      - *Nessuna Dipendenza Bloccante*: Risolte tutte le dipendenze con 
        le istruzioni precedenti.
]

=== Classificazione degli Stati dei Thread

// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Thread all'interno di un Warp")

  - Un warp contiene sempre 32 thread, ma non tutti potrebbero essere 
    logicamente attivi.
  - Lo stato di ogni thread è tracciato attraverso una *thread mask* o 
    *maschera di attività* (un registro hardware a 32 bit).
] 


// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Stati dei Thread")
  #figure(image("images/_page_42_Figure_8_2.2.jpeg", width: 70%))

  + Thread Attivo (Active Thread)
    - Esegue l'istruzione corrente del warp.
    - o Contribuisce attivamente all'esecuzione *SIMT*.
  + Thread Inattivo (Inactive Thread)
    - *Divergenza*: Ha seguito un percorso diverso nel warp per istruzioni 
      di controllo flusso, come salti condizionali.
    - *Terminazione*: Ha completato la sua esecuzione prima di altri thread nel warp.
    - *Padding*: I thread di padding sono utilizzati in situazioni in cui 
      il numero totale di thread nel blocco non è un multiplo di 32, per 
      garantire che il warp sia completamente riempito (puro overhead).
]

// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  *Gestione degli Stati*

  - Gli stati sono *gestiti automaticamente dall'hardware* attraverso maschere di esecuzione.
  - La transizione tra stati è dinamica durante l'esecuzione, quindi il numero di thread attivi può variare nel tempo. 
]



=== Scheduling dei Warp

// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Introduzione al Warp Scheduler")

  - Un'*unità hardware* presente in più copie all'interno di ogni 
    SM, responsabile della *selezione* e *assegnazione* dei warp 
    alle unità di calcolo CUDA.
  - *Obiettivo*: Massimizzare l'utilizzo delle risorse di calcolo 
    dell'SM, selezionando in modo efficiente i warp pronti e minimizzando 
    i tempi di inattività.
  - *Latency Hiding*: Contribuiscono a nascondere la latenza eseguendo 
    warp alternativi quando altri sono in stallo, garantendo un utilizzo 
    efficace delle risorse computazionali (prossime slide).

  #green_heading("Funzionamento Generale")
  - *Processo di Schedulazione*: I warp scheduler all'interno di un 
    SM #underline[selezionano i warp eleggibili] ad ogni ciclo di clock e #underline[li 
    inviano alle dispatch unit], responsabili dell'assegnazione 
    effettiva alle unità di esecuzione.
  - *Gestione degli Stalli*: Se un warp è in stallo, il warp scheduler 
    seleziona un altro warp eleggibile per l'esecuzione, garantendo 
    consentendo l'esecuzione continua e l'uso ottimale delle risorse di calcolo.
  - *Cambio di Contesto*: Il cambio di contesto tra warp è estremamente 
    rapido (on-chip per tutta la durata del warp) grazie alla partizione 
    delle risorse di calcolo e alla struttura hardware della GPU.
]

// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  *Limiti Architettonici*

  - Il numero di warp attivi su un SM è limitato dalle risorse 
    di calcolo. (Esempio: 64 warp concorrenti su un SM Kepler).
  - Il numero di warp selezionati ad ogni ciclo è limitato dal 
    numero di scheduler di warp. (Esempio: 4 su un SM Kepler).
]

#pagebreak()
==== Warp Scheduler e Dispatch Unit

// Griglia
#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  // Box sinistro
  
  // GreenBlock
  {block(
    //fill: rgb("#F5F9E8"),
    stroke: 1pt + light_green,
    radius: 0.8em,
    inset: 1.5em,
    width: 100%,
    // height: 21em,
    breakable: false
  )[
    #green_heading("Warp Scheduler")

    - È il "*cervello strategico*" che decide *#underline[quali]* warp mandare in esecuzione.
    - *Monitora* continuamente lo *stato dei warp* per identificare quelli eleggibili.
    - Gestisce la *priorità* e l'*ordine* di esecuzione dei warp, 
      cercando di minimizzare le latenze (_latency hiding_).
  ]
  // GreenBlock
  block(
    //fill: rgb("#F5F9E8"),
    stroke: 1pt + light_green,
    radius: 0.8em,
    inset: 1.5em,
    width: 100%,
    // height: 21em,
    breakable: false
  )[
    #green_heading("Dispatch Unit")
    - È il "*braccio esecutivo*" che si occupa di *come* eseguire i warp selezionati.
    - Si occupa di:
      - *Decodificare le istruzioni* del warp.
      - *Distribuire i thread* del warp alle unità di calcolo 
        appropriate (es. FP, INT, Tensor Cores).
      - *Recuperare i dati* dai registri e dalla memoria necessaria 
        per l'esecuzione.
      - *Assegnare fisicamente le risorse* hardware (registri, 
        unità di calcolo) ai thread.

  ]},
  //Box destro
  figure(image("images/_page_44_Figure_13_2.2.jpeg"))
)


==== Scheduling dei Warp: TLP e ILP
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Thread-Level Parallelism (TLP)")

  - *Definizione*: Esecuzione simultanea di più warp per sfruttare 
    il parallelismo tra thread.
  - *Funzionamento*: Quando un warp è in attesa (ad esempio, per 
    completare un'istruzione), un altro warp viene selezionato ed 
    eseguito, aumentando l'occupazione delle unità di calcolo.

  #green_heading("Instruction-Level Parallelism (ILP)")

  - *Definizione*: Esecuzione di istruzioni indipendenti all'interno 
    dello stesso warp.
  - *Funzionamento:* Se ci sono più istruzioni pronte da eseguire 
    in un warp, il warp scheduler può emettere queste istruzioni in 
    parallelo alle unità di esecuzione, massimizzando l'utilizzo 
    delle risorse (pipelining).
]

// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Importanza di TLP e ILP")

  - *Massimizzazione delle Risorse*: TLP e ILP contribuiscono a 
    mantenere le unità di calcolo attive e occupate, riducendo i 
    tempi morti durante l'esecuzione.
  - *Nascondere la Latenza*: TLP e ILP insieme aiutano a nascondere la 
    latenza delle operazioni di memoria e di calcolo, migliorando le 
    prestazioni complessive del sistema (vedi _latency hiding_).
]


=== Esecuzione Parallela dei Warp - Esempio con Fermi SM

// Griglia
#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  // Box sinistro
  {
    block(
      //fill: rgb("#F5F9E8"),
      stroke: 1pt + light_green,
      radius: 0.8em,
      inset: 1.5em,
      width: 100%,
      // height: 21em,
      breakable: false
    )[
      #set text(size: 8pt)
      #text(fill: light_green, weight: "bold")[Componenti Chiave per il Parallelismo]

      - *Due Scheduler di Warp*: Selezionano due warp pronti da 
        eseguire dai thread block assegnati all'SM.
      - *Due Unità di Dispatch delle Istruzioni*: Inviano le 
        istruzioni dei warp selezionati alle unità di esecuzione.

      #text(fill: light_green, weight: "bold")[Flusso di Esecuzione]

      - I blocchi vengono assegnati all'SM e *divisi in warp*.
      - Due scheduler selezionano warp *pronti* per l'esecuzione.
      - Ogni dispatch unit invia un'istruzione per warp a 16 
        CUDA Core, 16 unità di caricamento/memorizzazione (LD/ST), 
        4 unità di funzioni speciali (SFU).
      - Questo processo *si ripete ciclicamente*, consentendo 
        l'esecuzione parallela di più warp da più blocchi.

    ]

    block(
      fill: rgb("#F3F3F3"),
      radius: 0.8em,
      inset: 1.5em,
      width: 100%,
      // height: 21em,
      breakable: false
    )[
      #set text(size: 8pt)
      *Capacità*

      Fermi (compute capability 2.x) può gestire simultaneamente *48 warp* 
      per SM, per un totale di *1.536 thread residenti* in un singolo SM. 
      Ad ogni ciclo, al più *2 selected warps*.
    ]
  },
  {
    v(5em)
    image("images/_page_46_Figure_12_2.2.jpeg")
    dashed-box(
      black,
      "Poiché le risorse di calcolo sono partizionate tra i warp e mantenute *on-chip* durante l'intero ciclo di vita del warp, il cambio di contesto tra warp è immediato."
    )
  }
)


=== Scheduling Dinamico dell Istruzioni - Fermi SM
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  - Ad ogni ciclo di clock, un warp scheduler *emette un'istruzione* 
    pronta per l'esecuzione.
  - L'istruzione può provenire *dallo stesso warp* (ILP), #underline[se indipendente], 
    o più spesso da un warp diverso (TLP).
  - Se le risorse sono occupate, lo scheduler passa a un altro 
    warp pronto (_latency hiding_).

]

#figure(image("images/_page_47_2.2.png"))

=== Latency, Throughput e Concurrency
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  - #green_heading("Mean Latency:") La latenza media è la *media delle latenze* degli elementi individuali. La latenza di un singolo elemento è la differenza tra il suo tempo di inizio e il suo tempo di fine.
  - #green_heading("Throughput:") Il throughput rappresenta la velocità di elaborazione. È definito come il *numero di elementi completati entro un dato intervallo di tempo* diviso per la durata dell'intervallo stesso.
  - #green_heading("Concurrency:") La concurrency misura *quanti elementi vengono processati contemporaneamente* in un determinato momento. Si può definire sia istantaneamente che come media su un intervallo di tempo.
]
#figure(image("images/_page_48_Figure_4_2.2.jpeg"))

=== Latency Hiding nelle GPU
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Cosa è il Latency Hiding?")

  - Una tecnica che permette di *mascherare i tempi di attesa* dovuti ad operazioni ad alta latenza (come gli accessi alla memoria globale) attraverso l'esecuzione concorrente di più warp all'interno di un SM.
  - Si ottiene *intercambiando la computazione tra warp*, per massimizzare l'uso delle unità di calcolo di ogni SM.

  #green_heading("Funzionamento")

  - Ogni SM può gestire decine di warp concorrentemente da più blocchi (vedi compute capability della GPU).
  - Quando un warp è in stallo (es. accesso memoria), l'SM passa immediatamente all'esecuzione di altri warp pronti.
  - I Warp Scheduler dell'SM selezionano costantemente (ad ogni ciclo 
    di clock) i warp pronti all'esecuzione (#underline[occorre che abbiano sempre 
    warp eleggibili ad ogni ciclo]).

  #green_heading("Vantaggi del Latency Hiding")

  - *Migliore Utilizzo delle Risorse*: Le unità di elaborazione della GPU sono mantenute costantemente attive.
  - *Maggiore Throughput*: Completamento di un maggior numero di operazioni nello stessa unità di tempo.
  - *Minore Latenza Effettiva*: Minimizza l'impatto delle operazioni ad alta latenza.
]

// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  *Tipi di Latenza* (variano a seconda dell'architettura e dalla tipologia di operazione)

  - *Latenza Aritmetica*: Tempo di completamento di operazioni matematiche (bassa, es. 4-20 cicli).
  - *Latenza di Memoria*: Tempo di accesso ai dati in memoria (alta, es. 400-800 cicli per la memoria globale).
]
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Meccanismo dei Warp Scheduler")

  - L'immagine mostra *due warp scheduler* che gestiscono l'esecuzione di diversi warp nel tempo.
  - Warp Scheduler 0 e 1 *alternano l'esecuzione di warp diversi* per mantenere le unità di elaborazione occupate.
  - Quando un warp è in attesa (es. Warp 0 all'inizio), *altri warp vengono eseguiti* per nascondere la latenza.
  - I periodi di inattività (es. 'nessun eligible warp da eseguire') sono *minimizzati.*
  - Questo approccio permette di *mascherare i tempi di latenza* e aumentare l'efficienza complessiva.
  - Risorse pienamente utilizzate quando ogni scheduler ha un warp eleggibile ad *ogni ciclo di clock*.
]
#figure(image("images/_page_50_Figure_8_2.2.jpeg"))

=== Legge di Little

// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Cos'è la Legge di Little?")

  - La Legge di Little (dalla teoria delle code) ci aiuta a calcolare *quanti 
    warp (#underline[approssimativamente]) devono essere in esecuzione 
    concorrente* per ottimizzare il latency hiding e mantenere le unità 
    di elaborazione della GPU occupate. 
    #align(center)[#green_heading(`Warp Richiesti = Latenza x Throughput`)]

    - *Latenza:* Tempo di completamento di un'istruzione (in cicli di clock).
    - *Throughput:* Numero di warp (e, quindi, di operazioni) eseguiti per ciclo di clock.
    - *Warp Richiesti:* Numero di warp attivi necessari per nascondere la latenza.
  
  - Indica che per nascondere la latenza, è necessario avere un *numero sufficiente di warp in esecuzione o pronti per l'esecuzione*, in modo che mentre uno è in attesa, altri possano essere eseguiti.
]
// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  *Note*

  - La *latenza* e il *throughput* possono variare a seconda dell'architettura della GPU e del tipo di istruzioni.
  - Questa è una *stima*, il numero effettivo di warp necessari potrebbe essere leggermente diverso.
]


// Griglia
#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  // Box sinistro
  {
    // GreenBlock
    block(
      //fill: rgb("#F5F9E8"),
      stroke: 1pt + light_green,
      radius: 0.8em,
      inset: 1.5em,
      width: 100%,
      // height: 21em,
      breakable: false
    )[
      #green_heading("Esempio della Legge di Little")

      - *Latenza*: 5 Cicli
      - *Throughput desiderato*: 6 warp/ciclo

      Numero di Warp Richiesti = 5 x 6 = 30 warp in-flight.

      In questo caso, per mantenere un throughput di 6 warp/ciclo 
      con una latenza di 5 cicli, avremmo bisogno di almeno 30 warp 
      in esecuzione o pronti per l'esecuzione.
    ]
    // GrayBlock
    block(
      fill: rgb("#F3F3F3"),
      radius: 0.8em,
      inset: 1.5em,
      width: 100%,
      // height: 21em,
      breakable: false
    )[
      *Nota*: Un warp (32 thread) che esegue un'istruzione corrisponde 
      a *32 operazioni*  (1 operazione per thread)
    ]
  },
  //Box destro
  image("images/_page_52_Picture_7_2.2.jpeg")
)


=== Massimizzare il Parallelismo per Operazioni Aritmetiche

#block(
  width: 100%,
  fill: my_gray,
  radius: 8pt,
  inset: 15pt,
)[
  // Helper per lo stile
  #set text(size: 8pt)
  #let arch(name) = text(font: "Liberation Mono", weight: "bold", fill: nvidia-green, name)
  #let num(val) = text(font: "Liberation Mono", val)

  #table(
    columns: (auto, auto, auto, auto),
    // Allineamento centrato per tutte le celle
    align: center + horizon,
    stroke: none,
    column-gutter: 2em,
    row-gutter: -0.8em,

    // --- Intestazioni ---
    [*Architettura*],
    [*Latenza Istruzione*\ *(Cicli)*],
    [*Throughput*\ *(Operazioni/Ciclo)*],
    [*Parallelismo Necessario*\ *(Operazioni)*],

    // --- Riga 1: Fermi ---
    arch("Fermi"),
    num("20"),
    num("32 (1 warp/ciclo)"),
    num("640 (20 warp)"),

    // --- Riga 2: Kepler ---
    arch("Kepler"),
    num("20"),
    num("192 (6 warp/ciclo)"),
    // Composizione manuale per la parte rossa
    [
      #num("3,840 (")#text(fill: c-red, font: "Liberation Mono")[120 warp]#num(")")
    ]
  )
]

// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Esempio: Operazione Multiply-Add a 32-bit Floating-Point (a + b ") 
  #green_heading($times$) #green_heading(" c)")

  Limite Warp/SM in Kepler è 64

  Consideriamo una GPU con architettura Fermi:

  - *Throughput*: 32 operazioni/ciclo/SM
    - Un singolo SM può eseguire 32 operazioni di multiply-add a 32-bit 
      floating-point per ciclo di clock.
  - *Warp Richiesti per SM*: $640 div 32 "(operazioni per warp)" = 20 "warp"\/"SM" $
    - Per raggiungere il throughput massimo e per mantenere il pieno 
      utilizzo delle risorse computazionali, sono necessari 20 warp 
      attivi contemporaneamente su ogni SM.
]
// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  Esistono due modi principali per aumentare il parallelismo:

  - *ILP (Instruction-Level Parallelism)*: Aumentare il numero di istruzioni indipendenti all'interno di un singolo thread.
  - *TLP (Thread-Level Parallelism)*: Aumentare il numero di thread (e quindi di warp) che possono essere eseguiti contemporaneamente.
]
=== Massimizzare il Parallelismo per Operazioni di Memoria


#block(
  width: 100%,
  fill: my_gray,
  radius: 8pt,
  inset: 15pt,
)[
  // Helper per lo stile
  #set text(size: 8pt)
  #let arch(name) = text(font: "Liberation Mono", weight: "bold", fill: nvidia-green, name)
  #let num(val) = text(font: "Liberation Mono", val)

  #table(
    columns: (auto, auto, auto, auto, auto),
    // Allineamento centrato per tutte le celle
    align: center + horizon,
    stroke: none,
    column-gutter: 2em,
    row-gutter: -0.8em,

    // --- Intestazioni ---
    [*Architettura*],
    [*Latenza*\ *(Cicli)*],
    [*Bandwidth*\ *(GB/s)*],
    [*Bandwidth*\ *(B/ciclo)*],
    [*Parallelismo*\ *(KB)*],

    // --- Riga 1: Fermi ---
    arch("Fermi"),
    num("800"),
    num("144"),
    num("92"),
    num("74"),

    // --- Riga 2: Kepler ---
    arch("Kepler"),
    num("800"),
    num("250"),
    num("96"),
    num("77")
  )
]

// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Esempio: Operazione di Memoria")

  Consideriamo sempre una GPU con *architettura Fermi*:

  - *Calcolo del Bandwidth in Bytes/Ciclo:*
    -$ 144 "GB"\/s div 1.566 "GHz" approx 92 "Bytes"\/"Ciclo"$ (#underline[Frequenza di 
      memoria] Fermi -Tesla C2070 = 1.566 GHz)
  - *Calcolo del Parallelismo Richiesto:*
    - #green_heading("Parallelismo") = #green_heading("Bandwidth") (B/ciclo) 
      $times$ #green_heading("Latenza Memoria") (cicli)
    - Fermi: 92 B/ciclo $times$ 800 cicli ≈ 74 KB di I/O in-flight 
      per saturare il bus di memoria.
  - Memory Bandwidth è relativo all'intero device

  - *Interpretazione:*
    - 74 KB di operazioni di memoria necessarie per nascondere la latenza 
      (per l'intero device, non per SM).
      -  Memory Bandwidth è relativo all'intero device
]
#green_heading("Recuperare la Memory Frequency di una GPU NVIDIA (da terminale)")

```sh
$ nvidia-smi -a -q -d CLOCK | fgrep -A 3 "Max Clocks" | fgrep "Memory"
```

// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Esempio: Operazione di Memoria")

  - Il legame tra questi valori e il numero di warp/thread *varia 
    a seconda della specifica applicazione*.
  - *Conversione in Thread/Warp* (Supponendo 4 bytes ad esempio, FP32 per thread):
    - 74 KB $div$ 4 bytes/thread $approx$ 18,500 thread
    - 18,500 thread  $div$  32 thread/warp  $approx$  579 warp
    - Per 16 SM: 579 warp $div$ 16 SM = 36 warp per SM
  - Ovviamente, se ogni thread eseguisse più di un caricamento indipendente da 4 byte o un tipo di dato più grande (es. FP64), sarebbero necessari meno thread per mascherare la latenza di memoria.
]
// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  Esistono due modi principali per aumentare il parallelismo di memoria:

  - *Maggiore Granularità*: Spostare più dati per thread (ad esempio, 
    caricare più float per thread).
  - *Più Thread Attivi*: Aumentare il numero di thread concorrenti per 
    aumentare il numero di warp attivi.
]

=== Warp Divergence

// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Cosa è la Warp Divergence?")

  - In un warp, idealmente tutti i thread eseguono la *stessa istruzione contemporaneamente* per massimizzare il parallelismo SIMT (condividono un unico *Program Counter* [Architetture Pre-Volta] ).
  - Tuttavia, se un'*istruzione condizionale* (come un `if-else` o `switch`) porta thread diversi a percorrere *rami diversi* del codice, si verifica la *Warp Divergence.*
  - In questo caso, il warp esegue *serialmente ogni ramo*, utilizzando 
    una *maschera di attività* (calcolata #underline[automaticamente] in hardware) 
    per abilitare/disabilitare i thread.
  - La divergenza termina quando i thread *riconvergono* alla fine del costrutto condizionale.
  - La Warp Divergence *può significativamente degradare le prestazioni* perché i thread non vengono eseguiti in parallelo durante la divergenza (le risorse non vengono pienamente utilizzate).
  - Notare che il fenomeno della divergenza occorre *solo all'interno di un warp*.
]

// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
Esempio

```
if (threadIdx.x % 2 == 0) {
 // Istruzioni per thread con indice pari
} else {
 // Istruzioni per thread con indice dispari
}
```
]
=== CPU vs GPU: Gestione del Branching e della Warp Divergence

#block(
  width: 100%,
  fill: my_gray,
  radius: 8pt,
  inset: 15pt,
  breakable: false,
)[
  #grid(
    columns: (auto, 1fr, 1fr),
    column-gutter: 1.5em,
    row-gutter: 0.8em,
    
    // --- Intestazione ---
    [], // Cella vuota sopra la prima colonna
    align(center, text(fill: device-blue, weight: "bold", size: 1.1em)[CPU]),
    align(center, text(fill: nvidia-green, weight: "bold", size: 1.1em)[GPU]),

    // --- Riga 1: Esecuzione ---
    [*Esecuzione*],
    [Singoli thread o piccoli gruppi, *indipendenti* tra loro.],
    [Warp che eseguono le *stesse istruzioni* concorrentemente.],

    dashed-line,

    // --- Riga 2: Branch Prediction ---
    [*Branch Prediction*],
    [*Hardware dedicato*, con algoritmi di predizione complessi.],
    [*Non supportata*],

    dashed-line,

    // --- Riga 3: Esecuzione Speculativa ---
    [*Esecuzione Speculativa*],
    [Esegue istruzioni *in anticipo* basandosi sulla *branch prediction*.],
    [*Non supportata*],

    dashed-line,

    // --- Riga 4: Impatto della Divergenza ---
    [*Impatto della Divergenza*],
    [Mitigato dalla *branch prediction* e dall'*esecuzione speculativa*.],
    [Causa la *warp divergence*, riducendo il parallelismo e le prestazioni.],

    dashed-line,

    // --- Riga 5: Gestione della Divergenza ---
    [*Gestione della Divergenza*],
    [*Predizione* del ramo più probabile e *esecuzione speculativa* del codice.],
    [*Esecuzione seriale* dei rami divergenti nel warp, *disabilitando* i thread inattivi.],

    dashed-line,

    // --- Riga 6: Ottimizzazioni ---
    [*Ottimizzazioni*],
    [Meno critiche, gestite in parte *dall'hardware*.],
    [*Branch predication* (non lo vedremo) e *riorganizzazione del codice* essenziali.],

    dashed-line,

    // --- Riga 7: Considerazioni ---
    [*Considerazioni*],
    [Il costo della predizione errata è *relativamente basso*.],
    [Il costo della warp divergence è *elevato* a causa della perdita di parallelismo e dell'overhead di esecuzione seriale.],
  )
]

==== Warp Divergence: Analisi del Flusso di Esecuzione

#figure(image("images/_page_58_Figure_1_2.2.jpeg"))

// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  *Flusso*

  - All'inizio, tutti i thread eseguono lo stesso codice 
    ((#text(fill: blue)[*blocchi blu*]).
  - Quando si incontra un'*istruzione condizionale* 
    (#text(fill: orange)[*blocchi arancioni*]), il warp si *divide*.
  - Alcuni thread eseguono la clausola "#text(fill: nvidia-green)[*then*]"
    (#text(fill: nvidia-green)[*blocchi verdi*]), 
    mentre altri sono in #text(fill: rgb("#674EA7"))[*stallo*] 
    (#text(fill: rgb("#674EA7"))[*blocchi viola*]).
  - Nei momenti di divergenza, l'efficienza può scendere al 50% 
    (in questo caso, 16 thread attivi su 32).
]
==== Serializzazione nella Warp Divergence
// Griglia
#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  // Box sinistro
  // GreenBlock
  block(
    //fill: rgb("#F5F9E8"),
    stroke: 1pt + light_green,
    radius: 0.8em,
    inset: 1.5em,
    width: 100%,
    // height: 21em,
    breakable: false
  )[
    #green_heading("Divergenza")

    Quando i thread di un warp seguono percorsi diversi a causa di istruzioni condizionali (es. *if*), il warp esegue ogni ramo in serie, *disabilitando i thread inattivi*.

    #green_heading("Località")

    - La divergenza si verifica solo all'interno di un *singolo warp*.
    - Warp diversi operano *indipendentemente*.
    - I passi condizionali in *differenti warp* non causano divergenza.

    #green_heading("Impatto")

    - La divergenza può ridurre il parallelismo *fino a 32 volte.*
  ],
  //Box destro
  image("images/_page_59_Picture_9_2.2.jpeg")
)




#codly(header: [#align(center)[*Caso Peggiore*]])

```cpp
__global__ void WorstDivergence(int* x) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  switch (i % 32) {
    case 0 :
      x[i] = a(x[i]);
      break;
    case 1 :
      x[i] = b(x[i]);
      break;
    . . .
    case 31:
      x[i] = v(x[i]);
      break;
  }
}
```

Le prestazioni diminuiscono con l'aumento della divergenza nei warp.

==== Confronto delle Condizioni di Branch

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  // Box sinistro
  // GreenBlock
  {
    // GreenBlock
    block(
      //fill: rgb("#F5F9E8"),
      stroke: 1pt + light_green,
      radius: 0.8em,
      inset: 1.5em,
      width: 100%,
      // height: 21em,
      breakable: false
    )[
      #codly(header: [#align(center)[*Kernel 1*]])
      ```cpp
      __global__ void mathKernel1(float *sum) {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      float a = 0.0f, b = 0.0f;
      if (tid % 2 == 0) a = 100.0f;
      else b = 200.0f;
      *sum += a + b;} // Race condition (risolveremo con atomicAdd dopo)
      ```
    ]
    // GrayBlock
    block(
      fill: rgb("#F3F3F3"),
      radius: 0.8em,
      inset: 1.5em,
      width: 100%,
      // height: 21em,
      breakable: false
    )[
        #set text(size: 8pt)
        #green_heading("Funzionamento")

        Valuta la parità dell'*ID* di ogni singolo thread.

        #green_heading("Effetto sui thread")

        - *Thread pari* (ID 0, 2, 4, ...): eseguono il ramo *if*.
        - *Thread dispari* (ID 1, 3,...): eseguono il ramo *else*.

        #green_heading("Impatto sul warp")

        In ogni warp (32 thread), 16 thread eseguono *if* e 16 eseguono *else*.

        #green_heading("Risultato")

        *Warp divergence*, con esecuzione serializzata dei due percorsi all'interno del warp.

    ]
  },
    {
    // GreenBlock
    block(
      //fill: rgb("#F5F9E8"),
      stroke: 1pt + light_green,
      radius: 0.8em,
      inset: 1.5em,
      width: 100%,
      // height: 21em,
      breakable: false
    )[
      #codly(header: [#align(center)[*Kernel 2*]])
      ```cpp
      __global__ void mathKernel2(float *sum) {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      float a = 0.0f, b = 0.0f;
      if ((tid / warpSize) % 2 == 0) a = 100.0f;
      else b = 200.0f;
      *sum += a + b;}// Race condition (risolveremo atomicAdd dopo)
      ```
    ]
    // GrayBlock
    block(
      fill: rgb("#F3F3F3"),
      radius: 0.8em,
      inset: 1.5em,
      width: 100%,
      // height: 21em,
      breakable: false
    )[
        #set text(size: 8pt)
        #green_heading("Funzionamento")
        - tid \/ *warpSize*: Identifica l'*ID del warp* a cui appartiene il thread.
        - (...) % 2: Determina la parità del numero del warp.

        #green_heading("Effetto sui warp")

        - *Warp pari*: tutti i 32 thread eseguono il ramo *if*.
        - *Warp dispari*: tutti i 32 thread eseguono il ramo *else*.

        #green_heading("Impatto sul warp")

        Tutti i thread in un warp eseguono lo *stesso percorso.*

        #green_heading("Risultato")

        *Eliminazione del warp divergence*, con esecuzione parallela 
        all'interno di ogni warp (nessun overhead).

    ]
  }
)

// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Branch Efficiency") (calcolata in #green_heading("Nsight Compute"))

  La *Branch Efficiency* misura la percentuale di branch non divergenti rispetto al totale dei branch eseguiti da un warp.

#math.equation(
        numbering: none,
        block: true, 
        $ "Branch Efficiency" = 100 times ( (\# "Branches" - \# "DivergentBranches") 
          / (\# "Branches") )$
      )


  - Un *valore elevato* indica che la maggior parte dei branch eseguiti dal warp non causa divergenza.
  - Un *valore basso* indica un'elevata divergenza, con conseguente perdita di prestazioni.

  ```
  mathKernel1: Branch Efficiency 80.00%
  mathKernel2: Branch Efficiency 100.00%
  ```

  *Nota*: Nonostante la warp divergence, il compilatore CUDA applica ottimizzazioni anche con *-G* abilitato, risultando in una branch efficiency di *`mathKernel1`* superiore al 50% teorico.
]
=== Architetture Pre-Volta (< CC 7.0)
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #image("images/_page_63_Figure_10_2.2.png")

  - *Singolo Program Counter* e *Call Stack* condiviso per tutti i 
    32 thread del warp (puntano alla stessa istruzione).
  - Il warp agisce come una #underline[unità di esecuzione coesa/sincrona] (stato dei 
    thread è tracciato a livello di warp intero).
  - *Maschera di Attività (Active Mask)* per specificare i 
    thread attivi nel warp in ciascun istante.
  - La maschera viene *salvata* fino alla riconvergenza del warp, poi 
    *ripristinata* per riesecuzione sincrona.

  #green_heading("Limitazioni")

  - Quando c'è divergenza, i thread che prendono branch diverse *perdono concorrenza* fino alla riconvergenza.
  - Possibili *deadlock* tra thread in un warp, se i thread dipendono l'uno dall'altro in modo circolare.
]

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  // Box sinistro
  // GreenBlock
  {green_heading("Esempio di Divergenza (Pseudo-Code)")
  ```cpp
  if (threadIdx.x < 4) { 
    A; 
    B; 
  } else { 
    X; 
    Y; 
  } 
  Z;
  ```},
  {
    v(3em)
    image("images/page_63_Figure_2_2.2.png")
  }
  
)

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  // Box sinistro
  // GreenBlock
  // GrayBlock
  block(
    fill: rgb("#F3F3F3"),
    radius: 0.8em,
    inset: 1.5em,
    width: 100%,
    // height: 21em,
    breakable: false
  )[
    #green_heading("Esempio di Potenziale Deadlock") 
    ```cpp
    if (threadIdx.x < 4) {
      A;
      waitOnB();
    } else {
      B;
      waitOnA();
    }
    ```
  ],
  {
    v(3em)
    image("images/_page_64_Figure_1_2.2.png")
  }
)


==== Architettura Volta (CC 7.0+) e Independent Thread Scheduling
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Concetto chiave")

  L'*Independent Thread Scheduling (ITS)* consente #underline[piena concorrenza 
  tra i thread], indipendentemente dal warp.
]
#figure(image("images/_page_65_Picture_3_2.2.jpeg"))

// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Stato di Esecuzione per Thread")

  - Ogni thread mantiene il *proprio stato di esecuzione*, inclusi program counter e stack di chiamate.
  - Permette di #underline[cedere l'esecuzione] a livello di *singolo thread* (non sono più obbligati a eseguire in lockstep).

  #green_heading("Attesa per Dati")

  - Un thread può attendere che un altro thread produca dati, *facilitando 
    la comunicazione e la sincronizzazione* tra di essi.
  
  #green_heading("Ottimizzazione della Pianificazione")
    - Un *ottimizzatore di scheduling* raggruppa i thread attivi dello 
      stesso warp in unità SIMT.
    - Così facendo, si mantiene l'alto throughput dell'esecuzione SIMT, 
      come nelle GPU NVIDIA precedenti.

  #green_heading("Flessibilità Maggiore")

  - I thread possono ora divergere e riconvergere indipendentemente con *granularità sub-warp*.
  - Apre a pattern di programmazione che erano impossibili o problematici 
    nelle architetture precedenti.
]
==== Confronto Pre-Volta vs Post-Volta

#figure(image("images/_page_66_Figure_1_2.2.jpeg"))

#figure(image("images/_page_67_Figure_1_2.2.jpeg"))

=== Introduzione di `__syncwarp` in Volta
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Scopo")

  - Introdotta dall'architettura Volta *per supportare l'ITS* e *migliorare la gestione della divergenza* dei thread.
  - Permette di *sincronizzare esplicitamente e riconvergere* i thread 
    #underline[all'interno di un warp].
  - *Blocca l'esecuzione* del thread corrente finché tutti i thread 
    specificati nella maschera non hanno raggiunto il punto di sincronizzazione.

  ```cpp
  void __syncwarp(unsigned mask=0xffffffff);
  ```

  #green_heading("Vantaggi")

  - *Evita comportamenti non deterministici* dovuti alla divergenza intra-warp.
  - Garantisce che tutti i thread del warp specificato *siano allineati prima di comunicare o accedere a dati condivisi*.
  - Abilita l'*esecuzione sicura di algoritmi a grana fine* (riduzioni, scambi in shared memory, operazioni cooperative).
]
// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Esempio di Utilizzo")

  ```cpp
  if (threadIdx.x < 16) {
    // Codice per i primi 16 thread
  } else {
    // Codice per gli ultimi 16 thread
  }
  __syncwarp(); // Sincronizza tutti i thread del warp qui
  ```

  *Quando è davvero necessaria* #green_heading(`__syncwarp`)?

  Dopo una divergenza:
  - ❌ È superflua se ogni thread lavora su dati privati, senza comunicare.
  - ✅ È necessaria se c'è comunicazione o dipendenza tra thread del warp.
]


=== Confronto Pre-Volta vs Post-Volta
// --- Blocco Principale ---
#block(
  width: 100%,
  fill: my_gray,
  radius: 8pt,
  inset: 15pt,
  breakable: false,
)[
  #grid(
    columns: (auto, 1fr, 1fr),
    column-gutter: 1.5em,
    row-gutter: 0.8em,
    
    // --- Intestazione ---
    [], // Cella vuota sopra la prima colonna
    align(center, text(fill: nvidia-green, weight: "bold", size: 1.1em)[Pre-Volta]),
    align(center, text(fill: nvidia-green, weight: "bold", size: 1.1em)[Post-Volta]),

    // --- Riga 1: Program Counter ---
    [*Program Counter*],
    [Singolo #underline[per warp]],
    [Individuale #underline[per thread]],

    dashed-line,

    // --- Riga 2: Scheduling ---
    [*Scheduling*],
    [#underline[Lockstep]: tutti i thread del warp eseguono insieme],
    [#underline[Indipendente]: ogni thread può progredire autonomamente],

    dashed-line,

    // --- Riga 3: Sincronizzazione ---
    [*Sincronizzazione*],
    [#underline[Implicita]: i thread sono automaticamente sincronizzati],
    [#underline[Esplicita]: richiede #text(font: "Liberation Mono", fill: nvidia-green, weight: "bold")[`__syncwarp`]],

    dashed-line,

    // --- Riga 4: Divergenza ---
    [*Divergenza*],
    [Serializzazione dei rami divergenti],
    [Esecuzione interlacciata dei rami possibile],

    dashed-line,

    // --- Riga 5: Deadlock Intra-Warp ---
    [*Deadlock Intra-Warp*],
    [Possibili in certi scenari],
    [Largamente mitigati],

    dashed-line,

    // --- Riga 6: Prestazioni con Divergenza ---
    [*Prestazioni con Divergenza*],
    [Penalità per serializzazione],
    [Penalità simile, nessun miglioramento intrinseco],

    dashed-line,

    // --- Riga 7: Complessità del Codice ---
    [*Complessità del Codice*],
    [Workaround necessari per certi algoritmi],
    [Implementazioni più naturali possibili (ma richiede gestione esplicita)],
  )
]

// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #align(center)[
    #text(weight: "extrabold", size: 13pt)[ITS: Limitazioni]
  ]
  - ITS non può esonerare gli sviluppatori da una programmazione 
    parallela impropria. Nessuno scheduling hardware può salvare 
    dal *livelock* (ovvero thread che sono Sche tecnicamente in esecuzione 
    ma non fanno progressi reali).
  - Il progresso è garantito *solo per i warp residenti* al momento. 
    I thread rimarranno in Sinc attesa infinita se il loro progresso 
    dipende da un warp che non lo è.
  - Non garantisce la riconvergenza, quindi le assunzioni relative alla 
    programmazione a Dive ssibile livello di warp potrebbero non essere 
    valide (usare esplicitamente #green_heading(`__syncwarp`)).
  - Bisogna prestare più attenzione per garantire il comportamento SIMD dei warp.
  - ITS introduce *overhead hardware* per la gestione indipendente di 
    program counter e Pres call stack per ogni thread, aumentando la 
    flessibilità ma richiedendo più risorse. 

]


== Sincronizzazione e Comunicazione
=== Sincronizzazione in CUDA - Motivazioni
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("1. Asincronia tra Host e Device")
  - *Comportamento di Base*: L'host e il device operano in modo *asincrono*.
  - Senza sincronizzazione, l'host potrebbe tentare di utilizzare risultati *non ancora pronti* o *modificare dati ancora in uso* dalla GPU.
]
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("2. Sincronizzazione tra Thread all'Interno di un Blocco")

  - *Comportamento di Base*: I thread all'interno di un blocco possono 
    eseguire in #underline[ordine arbitrario] e a #underline[velocità diverse].
  - Quando i thread dello stesso blocco necessitano di condividere dati (utilizzando, ad esempio, la shared memory) o coordinare le loro azioni, è necessaria una sincronizzazione esplicita.
  ]

// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("3. Coordinazione all'Interno dei Warp")

  - *Comportamento di Base*:
    - *Pre-Volta*: I thread all'interno di un warp eseguivano sempre la stessa istruzione contemporaneamente (modello SIMD).
    - *Post-Volta* (CUDA 9.0+): Introdotta l'esecuzione indipendente dei thread (ITS) nel warp.
  - Con l'esecuzione indipendente dei thread, la sincronizzazione esplicita diventa necessaria per garantire la *coerenza nelle operazioni intra-warp*.
]


=== Race Condition (Hazard)

// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Cos'è?")

  Una *race condition* si verifica quando più thread accedono *concorrentemente 
  (almeno uno in scrittura)* e in modo *non sincronizzato* alla #underline[stessa locazione 
  di memoria], causando *risultati imprevedibili ed errori*.

  #green_heading("Tipi di Race Condition (Noti anche nella pipeline dei processori)")

  - *Read-After-Write (RAW):* Un thread legge prima che un altro finisca di scrivere.
  - *Write-After-Read (WAR):* Un thread scrive dopo che un altro ha letto, invalidando il valore.
  - *Write-After-Write (WAW):* Più thread scrivono nella stessa locazione, rendendo il valore indeterminato.

  #green_heading("Perché si verificano?")

  - I thread in un blocco sono logicamente paralleli ma #underline[non sempre 
    fisicamente simultanei].
  - Sono eseguiti in warp che possono trovarsi in punti diversi del codice.
  - *Senza sincronizzazione*, l'ordine di esecuzione tra thread è *imprevedibile*.
]
// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Prevenzione delle Race Condition")

  - *All'interno di un Thread Block*
    - Utilizzare #green_heading(`__syncthreads()`) per sincronizzare i thread 
      e garantire la visibilità dei dati condivisi.
    - #green_heading(`__syncthreads()`) garantisce che il thread A legga 
      dopo che il thread B ha scritto.
  - *Tra Thread Block diversi:*
    - Non esiste sincronizzazione diretta. L'unico modo sicuro è terminare il kernel e avviarne uno nuovo.
]

=== Deadlock in CUDA
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Cos'è?")

  - Un *deadlock* (o stallo) in CUDA si verifica quando i thread di un 
    blocco si bloccano reciprocamente #underline[in attesa di sincronizzazioni o risorse 
    non raggiungibili], causando il *blocco permanente* dell'esecuzione del kernel.
  - Può insorgere in presenza di *sincronizzazioni condizionali* o *dipendenze* non gestite correttamente.

  #green_heading("Condizioni per il Deadlock")

  - *Sincronizzazione Condizionale:* Uso di #green_heading(`__syncthreads()`) 
    all'interno di condizioni (#text(fill: rgb("#7B30D0"))[*`if`*], 
    #text(fill: rgb("#7B30D0"))[*`else`*]), dove solo 
    una parte dei thread     del blocco raggiunge il punto di sincronizzazione.
  - *Dipendenze Circolari:* Situazioni in cui gruppi di thread attendono 
    reciprocamente il completamento di operazioni, creando un ciclo di 
    dipendenze irrisolvibile.
  - *Risorse Condivise:* Gestione non corretta dell'accesso alla memoria 
    condivisa o ad altre risorse comuni.
]
// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Prevenzione/Gestione del Deadlock")

  - *Sincronizzazione Completa*: Evitare #green_heading(`__syncthreads()`) nei rami condizionali divergenti; assicurarsi che tutti i thread del blocco raggiungano i punti di sincronizzazione.
  - *Ristrutturazione del Codice*: Rimuovere le dipendenze condizionali organizzando le operazioni in modo che tutti i thread completino una fase prima di passare alla successiva.
  - *Independent Thread Scheduling*: Con architetture #green_heading("Volta") e successive, i thread di un warp possono avanzare in modo più indipendente, grazie all'Independent Thread Scheduling ed alleviare il problema.
]

=== Sincronizzazione in CUDA
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  La sincronizzazione è il meccanismo che permette di *coordinare* l'esecuzione di thread paralleli e garantire la *correttezza* dei risultati, evitando *race condition* *deadlock* e *accessi concorrenti non sicuri* alla memoria.
]
// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Livelli di Sincronizzazione in CUDA")

  - *Livello di Sistema (Host-Device)*:
    - #underline[Blocca l'applicazione host] finché tutte le operazioni sul device non sono completate.
    - Garantisce che il device abbia terminato l'esecuzione (copie, kernels, etc) prima che l'host proceda.
    - *Firma*: `cudaError_t cudaDeviceSynchronize(void); // può causare overhead bloccando l'host`
  - *Livello di Blocco (Thread Block)*:
    - Sincronizza tutti i thread all'interno di un singolo thread block.
    - Ogni thread attende che tutti gli altri thread nel blocco raggiungano il punto di sincronizzazione.
    - Garantisce la visibilità delle modifiche alla shared memory tra i thread del blocco.
    - *Firma*: `__device__ void __syncthreads(void); // riduce le prestazioni se usato troppo`
  - *Livello di Warp* (Disponibile con ITS a partire da CUDA 9.0 e architetture Volta+)
    - Sincronizza i thread all'interno di un #underline[singolo warp].
    - Garantisce la *riconvergenza* dei thread in presenza di divergenza.
    - *Ottimizza la cooperazione* tra thread dello stesso warp.
    - *Firma*: `__device__ void __syncwarp(unsigned mask=0xffffffff); // minimo overhead`
]
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  La sincronizzazione è il meccanismo che permette di *coordinare* l'esecuzione di thread paralleli e garantire la *correttezza* dei risultati, evitando *race condition*deadlock* e *accessi concorrenti non sicuri* alla memoria.
]
#green_heading("Esempi")

#image("images/_page_76_Figure_1_2.2.png")


=== Operazioni Atomiche in CUDA
v// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
#green_heading("Perché sono Necessarie le Operazioni Atomiche?")

- *Problema:* Race condition in operazioni *Read-Modify-Write*
  - Più thread *accedono e modificano* la stessa locazione di memoria contemporaneamente.
  - Risultati *imprevedibili* e *inconsistenti*.
]
#green_heading("Scenario Tipico")

```cpp
__global__ void increment(int *counter) {
  int old = *counter; // Legge il valore attuale dalla memoria
  old = old + 1; // Incrementa il valore letto
  *counter = old; // Scrive il nuovo valore nella stessa locazione
 }
```

// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
#green_heading("Conseguenze")

- *Conteggi Errati:* Il valore finale potrebbe non riflettere correttamente il numero di incrementi eseguiti.
- *Aggiornamenti di Dati Persi:* Le modifiche apportate da alcuni thread potrebbero essere sovrascritte da altri.
- *Comportamento Non Deterministico:* L'applicazione potrebbe dare risultati diversi ad ogni esecuzione.

#green_heading("Soluzione")

Operazioni atomiche *garantiscono l'integrità* delle operazioni Read-Modify-Write in ambiente concorrente.
]
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
#green_heading("Cosa sono le Operazioni Atomiche?")
(#link("https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html=atomic-functions")[Documentazione Online])

- Operazioni *Read-Modify-Write* eseguite (solo su funzioni device) come *singola istruzione hardware indivisibile*.

#green_heading("Caratteristiche")

- *Esclusività dell'accesso alla memoria*: L'hardware assicura che #underline[solo un 
  thread alla volta] può eseguire l'operazione #underline[sulla stessa locazione di memoria]. 
  I thread che eseguono operazioni atomiche sulla stessa posizione vengono 
  messi in coda ed eseguiti in serie (correttezza garantita).
- *Prevenzione delle interferenze*: Evitano che i thread interferiscano tra loro durante la modifica dei dati.
- *Compatibilità con memoria globale e condivisa*: Operano su word di 32, 64 bit o 128 bit.
- *Riduzione del parallelismo effettivo*, poiché i thread devono aspettare il proprio turno per accedere alla memoria.

#green_heading("Tipiche Operazioni Atomiche:")

- *Matematiche*: Addizione, sottrazione, massimo, minimo, incremento, decremento.
- *Bitwise*: Operazioni bit a bit come AND, OR, XOR.
- *Swap*: Scambio del valore in memoria con un nuovo valore.
]
#green_heading("Utilizzo di Base")

```cpp
__global__ void safeIncrement(int *counter) {
  atomicAdd(counter, 1); // Incrementa il valore atomico, evitando condizioni di gara
}
```

=== Operazioni Atomiche in CUDA - Esempi d'Uso

#codly(header: [#align(center)[*Operazioni Atomiche*]])
```cpp
// Operazioni atomiche aritmetiche
int atomicAdd(int* addr, int val); // Somma val a *addr
int atomicSub(int* addr, int val); // Sottrae val da *addr
int atomicMax(int* addr, int val); // Aggiorna *addr al max tra *addr e val
int atomicMin(int* addr, int val); // Aggiorna *addr al min tra *addr e val
unsigned int atomicInc(unsigned int* addr, unsigned int val); // Incrementa *addr, ciclato tra 0 e val
unsigned int atomicDec(unsigned int* addr, unsigned int val); // Decrementa *addr, ciclato tra 0 e val
// Operazioni atomiche di confronto
int atomicExch(int* addr, int val); // Scambia *addr con val
int atomicCAS(int* addr, int cmp, int val); // Confronta *addr con cmp, aggiorna *addr a val se uguali
// Operazioni atomiche bitwise
int atomicAnd(int* addr, int val); // AND tra *addr e val, aggiorna addr
int atomicOr(int* addr, int val); // OR tra *addr e val, aggiorna addr
int atomicXor(int* addr, int val); // XOR tra *addr e val, aggiorna addr
```

#block(
  stroke: (paint: black, dash: "dashed", thickness: 1pt),
  radius: 5pt,
  inset: 8pt,
  width: 100%
  )[
  - Leggono il valore originale dalla memoria, eseguono l'operazione e salvano 
    il risultato *nello stesso indirizzo*, *restituendo il valore originale pre-modifica*.
  - *Supporto a Tipi Estesi:* Esistono anche varianti atomiche per operazioni 
    su tipi a 64 bit (*`long` `long int`*) e floating point (*`float`* e *`double`*), 
    supportate su architetture recenti.
]

== Ottimizzazione delle Risorse
=== Resource Partitioning in CUDA
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
#green_heading("Cos'è il Resource Partitioning?")

- Come abbiamo visto, assegnare molti warp a un SM aiuta a nascondere la latenza, ma i limiti delle risorse possono impedire di raggiungere il massimo supportato.
- Il *Resource Partitioning* riguarda la *suddivisione e la gestione delle risorse hardware* limitate all'interno di una GPU, in particolare all'interno di ogni SM.
- L'obiettivo è *trovare un equilibrio* nella distribuzione di registri e memoria condivisa tra thread e blocchi, *ottimizzando l'efficienza complessiva dell'esecuzione* dei kernel CUDA.

#green_heading("Partizionamento delle Risorse nell'SM")

- Ogni SM ha una quantità limitata di registri e memoria condivisa:
  - *Register File*: Un insieme di registri a 32 bit, partizionati tra i thread attivi.
  - *Memoria Condivisa*: Una quantità fissa di memoria condivisa, partizionata tra i blocchi di thread attivi.
- Il numero di thread block e warp che possono risiedere simultaneamente su un SM dipende dalla:
  - *Disponibilità di Risorse*: Quantità di registri e memoria condivisa disponibili sull'SM.
  - *Richiesta del Kernel*: Quantità di registri e memoria condivisa richiesti dal kernel per l'esecuzione.
- Se le risorse di uno SM non permettono di eseguire almeno un blocco di thread, il kernel *fallisce*.
]
#pagebreak()

=== Anatomia di un Thread Block

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  // Box sinistro
  {
    block(
    fill: rgb("#F3F3F3"),
    radius: 0.8em,
    inset: 1.5em,
    width: 100%,
    // height: 21em,
    breakable: false
    )[
    #green_heading("Requisiti di Risorse per SM")

      Tutti i blocchi in una griglia eseguono lo stesso programma 
      usando lo stesso numero di thread, portando a *3 requisiti di 
      risorse fondamentali*:
    ]
    
    // GreenBlock
    block(
      //fill: rgb("#F5F9E8"),
      stroke: 1pt + light_green,
      radius: 0.8em,
      inset: 1.5em,
      width: 100%,
      // height: 21em,
      breakable: false
    )[
      #green_heading("1. Dimensione del Blocco")

      Il numero di thread che devono essere concorrenti.

      #green_heading("2. Memoria Condivisa")

      È comune a tutti i thread dello stesso blocco.

      #green_heading("3. Registri")

      Dipendono dalla complessità del programma.
      (*thread-per-blocco* $times$ *registri-per-thread*)
    ]

  },
  {
    image("images/_page_82_Picture_11_2.2.jpeg")
    text()[Un blocco mantiene un numero costante di thread ed esegue 
    unicamente su un singolo SM]

    ```cpp
    __global__ void simpleKernel(float* out) {
      int tid = threadIdx.x;
      float myValue = 3.14;
      __shared__ float sharedData[64];
      sharedData[tid] = myValue;
      __syncthreads();
      out[tid] = sharedData[tid] * myValue;
    }
    ```
  }


)

#let circled_color(color, body) = box(
  stroke: (paint: color, thickness: 1.4pt),
  radius: 50%, // Raggio al 50% crea un ovale perfetto
  inset: (x: 10pt, y: 8pt), // Padding orizzontale maggiore per allungare l'ovale
  body
)


#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  // Box sinistro
  // GreenBlock
  image("images/_page_83_Figure_2_2.2.png"),

  {
    align(center)[#text(weight: "bold", size: 7pt)[Thread Block]]
    image("images/_page_83_Picture_4_2.2.jpeg")
    text()[Un blocco contiene un numero fisso di thread ed esegue 
    unicamente su un singolo SM]
    // GrayBlock
    block(
      fill: rgb("#F3F3F3"),
      radius: 0.8em,
      inset: 1.5em,
      width: 100%,
      // height: 21em,
      breakable: false
    )[
      #align(center)[*Esempio di Requisiti di Risorse per i Blocchi*]
      #table(
        columns: (auto, auto),
        column-gutter: 1em,
        align: (left, center),
        stroke: none,

        [#blue_heading()[Thread per Blocco]], [`256`],
        [Registri per Thread], [`64`],
        [#blue_heading()[Registri per Blocco]],[`(256*64)=16384` ],
        [#blue_heading()[Shared Memory per Blocco]],[#circled_color(orange, `48Kb`)]

      )
    ]
  }
)

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  // Box sinistro
  // GreenBlock
  image("images/_page_84_Figure_2_2.2.jpeg"),

  {
    align(center)[#text(weight: "bold", size: 7pt)[Thread Block]]
    image("images/_page_83_Picture_4_2.2.jpeg")
    text()[Un blocco contiene un numero fisso di thread ed esegue 
    unicamente su un singolo SM]
    // GrayBlock
    block(
      fill: rgb("#F3F3F3"),
      radius: 0.8em,
      inset: 1.5em,
      width: 100%,
      // height: 21em,
      breakable: false
    )[
      #align(center)[*Esempio di Requisiti di Risorse per i Blocchi*]
      #table(
        columns: (auto, auto),
        column-gutter: 1em,
        align: (left, center),
        stroke: none,

        [#blue_heading()[Thread per Blocco]], [`256`],
        [Registri per Thread], [`64`],
        [#blue_heading()[Registri per Blocco]],[`(256*64)=16384` ],
        [#blue_heading()[Shared Memory per Blocco]],
        [#circled_color(nvidia-green, `32Kb`)]

      )
    ]
  }
)

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  image("images/_page_86_Figure_2_2.2.jpeg"),
  {
    // GrayBlock
    block(
      fill: rgb("#F3F3F3"),
      radius: 0.8em,
      inset: 1.5em,
      width: 100%,
      // height: 21em,
      breakable: false
    )[
      #align(center)[*Esempio di requisiti (Griglia Blue)*]
      #table(
        columns: (auto, auto),
        column-gutter: 1em,
        align: (left, center),
        stroke: none,

        [#blue_heading()[Thread per Blocco]], [`256`],
        [#blue_heading()[Registri per Thread]], [`64`],
        [#blue_heading()[Registri per Blocco]],[`(256*64)=16384` ],
        [#blue_heading()[Shared Memory per Blocco]],[`48Kb`]

      )
    ]
        block(
      fill: rgb("#F3F3F3"),
      radius: 0.8em,
      inset: 1.5em,
      width: 100%,
      // height: 21em,
      breakable: false
    )[
      #align(center)[*Esempio di requisiti (Griglia Arancione)*]
      #table(
        columns: (auto, auto),
        column-gutter: 1em,
        align: (left, center),
        stroke: none,

        [#orange_heading()[Thread per Blocco]], [`512`],
        [#orange_heading()[Registri per Thread]], [`32`],
        [#orange_heading()[Registri per Blocco]],[`(512*32)=16384` ],
        [#orange_heading()[Shared Memory per Blocco]],[`0`]

      )
    ]
  }
)

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  image("images/_page_87_Figure_2_2.2.jpeg"),
  {
    align(center)[#text(weight: "bold", size: 7pt)[Thread Block]]
    image("images/_page_83_Picture_4_2.2.jpeg")
    text()[Un blocco contiene un numero fisso di thread ed esegue 
    unicamente su un singolo SM]
    // GrayBlock
    block(
      fill: rgb("#F3F3F3"),
      radius: 0.8em,
      inset: 1.5em,
      width: 100%,
      // height: 21em,
      breakable: false
    )[
      #align(center)[*Esempio di Requisiti di Risorse per i Blocchi*]
      #table(
        columns: (auto, auto),
        column-gutter: 1em,
        align: (left, center),
        stroke: none,

        [#blue_heading()[Thread per Blocco]], 
        [#circled_color(light_orange, `768`)],
        [#blue_heading()[Registri per Thread]], [`16`],
        [#blue_heading()[Registri per Blocco]],[`(768*16)=12288` ],
        [#blue_heading()[Shared Memory per Blocco]],[`32Kb`]

      )
    ]
  }
)


#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  image("images/_page_88_Figure_2_2.2.jpeg"),
  {
    align(center)[#text(weight: "bold", size: 7pt)[Thread Block]]
    image("images/_page_83_Picture_4_2.2.jpeg")
    text()[Un blocco contiene un numero fisso di thread ed esegue 
    unicamente su un singolo SM]
    // GrayBlock
    block(
      fill: rgb("#F3F3F3"),
      radius: 0.8em,
      inset: 1.5em,
      width: 100%,
      // height: 21em,
      breakable: false
    )[
      #align(center)[*Esempio di Requisiti di Risorse per i Blocchi*]
      #table(
        columns: (auto, auto),
        column-gutter: 1em,
        align: (left, center),
        stroke: none,

        [#blue_heading()[Thread per Blocco]], 
        [#circled_color(nvidia-green, `1024`)],
        [#blue_heading()[Registri per Thread]], [`16`],
        [#blue_heading()[Registri per Blocco]],[`(1024*16)=12288` ],
        [#blue_heading()[Shared Memory per Blocco]],[`32Kb`]

      )
    ]
  }
)


==== Compute Capability (CC) - Limiti SM
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  - La *Compute Capability (CC)* di NVIDIA è un numero che identifica le 
    caratteristiche e le capacità di una GPU NVIDIA in termini di 
    #underline[funzionalità supportate] e #underline[limiti hardware].
  - È composta da *due numeri*: il numero principale indica la *generazione* 
    dell'architettura, mentre il numero secondario indica *revisioni* 
    e *miglioramenti* all'interno di quella generazione.
]


#block(
  width: 100%,
  fill: my_gray,
  radius: 8pt,
  inset: 10pt, // Inset ridotto leggermente per massimizzare lo spazio orizzontale
  breakable: false,
)[
  #set text(size: 8pt)
  // Nota in alto a destra
  #align(right)[\*Valori concorrenti per singolo SM]
  
  #v(0.5em)

  #table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    // Allinea tutte le celle al centro orizzontalmente e verticalmente
    align: center + horizon, 
    stroke: none,
    column-gutter: 0.5em, // Spazio tra le colonne
    row-gutter: -0.8em,    // Spazio tra le righe

    // --- Intestazioni ---
    // Uso le interruzioni di riga manuali (\) per replicare l'impaginazione dell'immagine
    hdr[Compute\ Capability],
    hdr[Architettura],
    hdr[Max Thread\ per Blocco],
    hdr[Max Thread\ per SM\*],
    hdr[Max Warps\ per SM\*],
    hdr[Max Blocchi\ per SM\*],
    hdr[Max Registri\ per Thread],
    hdr[Memoria Condivisa\ per SM],

    // --- Riga 1 ---
    num[1.x], arch[Tesla], num[512], num[768], num[24/32], num[8], num[124], num[16KB],
    
    // --- Riga 2 ---
    num[2.x], arch[Fermi], num[1024], num[1536], num[48], num[8], num[63], num[48KB],

    // --- Riga 3 ---
    num[3.x], arch[Kepler], num[1024], num[2048], num[64], num[16], num[255], num[48KB],

    // --- Riga 4 ---
    num[5.x], arch[Maxwell], num[1024], num[2048], num[64], num[32], num[255], num[64KB],

    // --- Riga 5 ---
    num[6.x], arch[Pascal], num[1024], num[2048], num[64], num[32], num[255], num[64KB],

    // --- Riga 6 ---
    num[7.x], arch[Volta/Turing], num[1024], num[1024/2048], num[32/64], num[16/32], num[255], num[96KB],

    // --- Riga 7 ---
    num[8.x], arch[Ampere/Ada], num[1024], num[1536/2048], num[48/64], num[16/24], num[255], num[164KB],

    // --- Riga 8 ---
    num[9.x], arch[Hopper], num[1024], num[2048], num[64], num[32], num[255], num[228KB],

    // --- Riga 9 ---
    num[10.x/12.x], arch[Blackwell], num[1024], num[2048/1536], num[64/48], num[32], num[255], num[128KB],
  )
]
#align(center)[#link("https://en.wikipedia.org/wiki/UDA=Version\_features\_and\_specifications")]



=== Occupancy
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
#green_heading("Cosa è l'Occupancy?")

  - L'occupancy rappresenta il *grado di utilizzo delle risorse* di calcolo dell'SM.
  - L'occupancy è il *rapporto* tra i warp attivi e il numero massimo di warp supportati per SM (vedi compute capability):

  #align(center)[`Occupancy [%] = Active Warps / Maximum Warps`] 

  #green_heading("Punti Chiave")

  - L'occupancy misura l'efficacia nell'uso delle risorse dell'SM:
    - *Occupancy Ottimale*: Quando raggiunge un livello sufficiente per nascondere la latenza. Un ulteriore aumento potrebbe degradare le prestazioni a causa della riduzione delle risorse disponibili per thread.
    - *Occupancy Bassa*: Risulta in una scarsa efficienza nell'emissione delle istruzioni, poiché non ci sono abbastanza warp eleggibili per nascondere la latenza tra istruzioni dipendenti.
  - *Un'occupancy elevata non garantisce sempre prestazioni migliori*: Oltre certa soglia, fattori come i pattern di accesso alla memoria e il parallelismo delle istruzioni possono diventare più rilevanti per l'ottimizzazione.
]
// GrayBlock
#block(
  fill: rgb("#F3F3F3"),
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Strumenti per l'Ottimizzazione")

  - *Strumenti di Profiling:* Nsight Compute consente di recuperare facilmente 
    l'occupancy, offrendo dettagli sul numero di warp attivi per SM e 
    sull'efficienza delle risorse di calcolo (tuttavia, l'occupancy non 
    deve #underline[mai essere guardata in isolamento]. Diventa utile se combinata 
    con altre metriche del profiler).
  - *Suggerimento*: Osservare gli effetti sul tempo di esecuzione del kernel a diversi livelli di occupancy.
]

==== Occupancy Teorica vs Effettiva
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Misure di Occupancy")

  L'occupancy di un kernel CUDA si divide in *teorica*, basata sui limiti hardware, ed *effettiva*, misurata a runtime.

  #green_heading("Occupancy Teorica (Theoretical)")

  - L'occupancy teorica è *determinata dalla configurazione di lancio* (numero di blocchi/thread, quantità di memoria condivisa, numero di registri per thread) e i limiti dell'SM (compute capability).
  - *Limite massimo warp attivi per SM* = (*Limite massimo blocchi attivi*) × (*Warp per blocco*)
  - È possibile aumentare il limite incrementando il numero di warp per blocco (dimensioni del blocco) o modificando i fattori limitanti (registri e/o shared memory) per aumentare i blocchi attivi per SM.

  #green_heading("Occupancy Effettiva (Achieved)")

  - Misura il *numero reale di warp attivi* durante l'esecuzione del kernel.
  - Il numero reale di warp attivi varia durante l'esecuzione del kernel, man mano che i warp iniziano e terminano.
  - *Calcolo dell'occupazione effettiva* (vedere Nsight Compute):
    - L'occupazione ottenuta è misurata su ciascun scheduler di warp utilizzando *contatori di prestazioni hardware* che registrano i warp attivi ad ogni ciclo di clock.
    - I conteggi vengono sommati su tutti i warp scheduler di ogni SM (1 per SMSP) e divisi per i cicli di clock attivi dell'SM per calcolare la *media dei warp attivi*.
    - Dividendo per il numero massimo di warp attivi supportati dall'SM (Maximum Warps), si ottiene l'*occupazione effettiva media* per SM durante l'esecuzione del kernel.
]
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Obiettivi di Ottimizzazione")

  - L'occupazione effettiva *non può* superare l'occupazione teorica (rappresenta il limite superiore).
  - Pertanto, il primo passo per aumentare l'occupazione è *incrementare quella teorica*, modificando i fattori limitanti.
  - Successivamente, è necessario verificare se il valore ottenuto è vicino a quello teorico per ridurre il gap.

  #green_heading("Cause di Bassa Occupazione Effettiva")

  - L'occupancy effettiva sarà inferiore a quella teorica quando il numero 
    teorico di warp attivi non viene mantenuto durante l'attività dello 
    SM (il problema forse non è il resource partitioning). Ciò può 
    accadere quando si ha:
    - *Carico di lavoro sbilanciato nei blocchi*: Quando i warp in un
      blocco hanno tempi di esecuzione diversi (es. warp divergence), 
      si crea un "*tail effect*" che riduce l'occupazione. Soluzione: 
      bilanciare il carico tra i warp.
    - *Carico di lavoro sbilanciato tra blocchi*: Se i blocchi della grid 
      hanno durate diverse, si può lanciare un maggior numero di blocchi o 
      kernel concorrenti per ridurre l'effetto coda.
    - *Numero insufficiente di blocchi lanciati*: Se la grid ha meno blocchi 
      del numero di SM del device, alcuni SM rimarranno inattivi. Ad esempio, 
      lanciare 60 blocchi su un dispositivo con 80 SM lascia 20 SM sempre inattivi, 
      riducendo l'utilizzo complessivo della GPU.
    - *Wave Parziale*: L'ultima ondata di blocchi potrebbe non saturare tutti 
      gli SM. Ad esempio, con 80 SM che supportano 2 blocchi ciascuno e una grid 
      di 250 blocchi: le prime due wave eseguono 160 blocchi (80 SM $times$ 2), ma 
      la terza wave ha solo 90 blocchi, lasciando alcuni SM parzialmente 
      utilizzati o inattivi.
]
==== Nota Importante sull'Occupancy
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Ricorda")

  L'obiettivo finale *non è massimizzare l'occupancy*, ma #underline[*minimizzare il 
  tempo di esecuzione del kernel*].
]
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #green_heading("Linee guida pratiche")

  - ❌*Occupancy < 25-30 % → problema serio*
    - Non ci sono abbastanza warp per nascondere le latenze (critico soprattutto per kernel memory-bound).
    - La GPU risulta sottoutilizzata, con SM spesso inattivi
    - *Azione*: ridurre l'uso di registri / shared memory per thread oppure aumentare il numero di thread per blocco, se possibile.
  - ✅*Occupancy 50-80 % → generalmente buono*
    - Warp sufficienti per un buon *latency hiding* nella maggior parte dei casi
    - Risorse adeguate per ciascun thread
    - *Focus*: ottimizzare coalescing, divergenza, e accessi alla memoria
  - ⚠️*Occupancy > 90 % → non sempre vantaggiosa* (kernel memory bound compute bound )
    - Latency hiding ottimo ma attenzione: se limita troppo le risorse il guadagno si perde.
    - Oltre una certa soglia, più occupancy potrebbe non migliorare le performance.
  - *Occupancy =* strumento per trovare il *giusto equilibrio tra latency hiding e risorse per thread*.
  ]

==== Nsight Compute: Occupancy Calculator
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  Nsight Compute offre uno strumento utile chiamato "Occupancy Calculator" (<u>Documentazione</u>) che consente di:

  - *Stimare l'Occupancy*: Calcola l'occupancy di un kernel CUDA su una determinata GPU.
  - *Ottimizzare le Risorse*: Mostra l'impatto di registri e memoria condivisa sull'occupancy.
  - *Migliorare le Prestazioni*: Fornisce suggerimenti per massimizzare l'uso delle risorse dell'SM e migliorare le prestazioni complessive.
]
#image("images/_page_94_Figure_5_2.2.jpeg")

#green_heading("Nsight Compute: Occupancy Calculator")
// GreenBlock
#block(
  //fill: rgb("#F5F9E8"),
  stroke: 1pt + light_green,
  radius: 0.8em,
  inset: 1.5em,
  width: 100%,
  // height: 21em,
  breakable: false
)[
  #align(center)[*Linee Guida per le Dimensioni di Griglia e Blocchi*]
  - Mantenere il numero di thread per block multiplo della dimensione del warp (32).
  - Evitare dimensioni di block piccole: Iniziare con almeno 128 o 256 thread per block.
  - 
]
   Regolare la dimensione del blocco in base ai requisiti di risorse del kernel. Mantenere il *numero di blocchi molto maggiore del numero di SM* per esporre sufficiente parallelismo al dispositivo (latency hiding). Condurre esperimenti per scoprire la migliore configurazione di esecuzione e utilizzo delle risorse. Impatto della Variazione dell'Uso della Memoria Condivisa Per Blocco

=== Panoramica del Modello di Esecuzione CUDA

- *Architettura Hardware GPU*
  - Introduzione al Modello di Esecuzione CUDA
  - Organizzazione degli Streaming Multiprocessors (SM)
  - Panoramica delle Architetture GPU NVIDIA
- *Organizzazione e Gestione dei Thread*
  - Mappatura tra Vista Logica e Hardware
  - Distribuzione e Schedulazione dei Blocchi sui SM
- *Modello di Esecuzione SIMT e Warp*
  - Confronto tra SIMD e SIMT
  - Warp e Gestione dei Warp
  - Latency Hiding e Legge di Little
  - Warp Divergence e Thread Independent Scheduling
- *Sincronizzazione e Comunicazione*
  - Meccanismi di Sincronizzazione
  - Operazioni Atomiche
- *Ottimizzazione delle Risorse*
  - Resource Partitioning
  - Occupancy
- *Parallelismo Avanzato*
  - CUDA Dynamic Parallelism

=== Introduzione al CUDA Dynamic Parallelism

#green_heading("Il Problema:")

- Algoritmi complessi (altamente dinamici) possono richiedere *strutture di parallelismo più flessibili.*
- La suddivisione dei problemi in kernel separati da lanciare in sequenza *dalla CPU* creano un collo di bottiglia.

#green_heading("La Soluzione: Dynamic Parallelism")

- Introdotto in CUDA 5.0 nel 2012 (Architettura Kepler), il CUDA Dynamic Parallelism (CDP) è disponibile su device con una Compute Capability 3.5 o superiore.
- Permette la *creazione* e *sincronizzazione* dinamica (on the fly) di nuovi kernel direttamente dalla GPU.
- È possibile posticipare a *runtime* la decisione su quanti blocchi e griglie creare sul device (utile quando la *quantità di lavoro nidificato è sconosciuta*)
- Supporta un approccio *gerarchico* e *ricorsivo* al parallelismo *evitando* continui passaggi fra CPU e GPU.

#green_heading("Possibili Applicazioni")

- *Algoritmi ricorsivi* (es: Quick Sort, Merge Sort) → [Ricorsione con profondità sconosciuta]
- *Strutture dati ad albero* (es: Alberi di ricerca, Alberi decisionali) → [Elaborazione parallela nidificata irregolare]
- *Elaborazione di immagini e segnali* (es. Region growing) → [Decomposizione dinamica delle aree di elaborazione]

#green_heading("Vantaggi")

- *Flessibilità*: Adattamento dinamico del parallelismo in base ai dati elaborati, senza dover prevedere tutto a priori.
- *Scalabilità*: Sfruttamento ottimale delle risorse GPU, creando nuovi blocchi e griglie solo quando necessario.
- *Efficienza*: Riduzione del collo di bottiglia CPU-GPU, spostando parte del controllo dell'esecuzione sulla GPU.

=== Dynamic Parallelism: Eliminare il Round-trip CPU-GPU

*Lancio da CPU (Approccio Tradizionale)*

```
__global__ void kernelA() {
}
__global__ void kernelB() {
}
int main() {
 ...
 kernelA<<<1,1>>>();
 ... // ottieni i risultati
 if(condition) {
 kernelB <<<1, 1>>>();
 }
}
```

*Lancio dalla GPU (Dynamic Parallelism)*

```
__global__ void kernelB() {
}
__global__ void kernelA() {
 if(condition)
 kernelB <<<1, 1>>>();
}
int main() {
 ...
 kernelA<<<1,1>>>();
}
```

=== Dynamic Parallelism: Eliminare il Round-trip CPU-GPU

#image("images/_page_99_Figure_1_2.2.jpeg")

#image("images/_page_99_Figure_2_2.2.jpeg")

=== Esecuzione Nidificata con CUDA Dynamic Parallelism

#green_heading("Come Funziona:")

- Un thread, un blocco di thread o una griglia (*parent*) lancia una nuova griglia (*child grid*).
- Una child grid lanciata con dynamic parallelism *eredita* dal kernel padre certi attributi e limiti come, ad esempio, la configurazione della *cache L1/memoria condivisa* e *dimensione dello stack*.
- I blocchi della griglia child possono essere eseguiti in parallelo e in modo indipendente rispetto al kernel padre.
- Il kernel/griglia parent continua immediatamente dopo il lancio del kernel child (*asincronicità*).
- Il *child deve sempre completare prima che il thread/blocco/griglia parent sia considerato completo.*
- Un parent si considera *completato* solo quando tutte le griglie child create dai suoi thread (tutti) hanno terminato l'esecuzione.

#green_heading("Visibilità e Sincronizzazione:")

- Ogni child grid lanciata da un thread è *visibile a tutti i thread dello stesso blocco*.
- Se i thread di un blocco terminano prima che tutte le loro griglie child abbiano completato, il sistema attiva automaticamente una *sincronizzazione implicita* per attendere il completamento di queste griglie.
- Un thread può *sincronizzarsi esplicitamente* con le proprie griglie child e con quelle lanciate da altri thread *nel suo blocco* utilizzando *primitive di sincronizzazione* (*cudaDeviceSynchronize*).
- Quando un thread parent lancia una child grid, *l'esecuzione della griglia figlio non è garantita immediatamente*, a meno che il blocco di thread genitore non esegua una *sincronizzazione esplicita*.

=== Esempio di CUDA Dynamic Parallelism

```
// Kernel Figlio
__global__ childKernel(void* data){
 // Operazioni sui dati
}
// Kernel Genitore
__global__ parentKernel(void *data){
 childKernel<<<16, 16>>>(data);
}
// Chiamata del Parent Kernel dall'Host
parentKernel<<<256, 64>>(data);
```

```
// Kernel ricorsivo supportato
__global__ recursiveKernel(void* data){
 if(continueRecursion == true)
 recursiveKernel<<<64, 16>>>(data);
}
```

========= Struttura del Codice

- *Stessa sintassi* usata nel codice host.
- Si noti che ogni thread che incontra un lancio di kernel *lo esegue*.
- Quanti thread vengono lanciati in totale per l'esecuzione di *childKernel*?

Nel caso in cui si desidera *solo una griglia child per blocco parent* usare:

```
if ( threadIdx.x == 0 )
 childKernel <<<16,16>>>(data);
```

*Configurazione Griglia/Blocco:* I kernel lanciati dinamicamente possono avere una configurazione di griglia e blocco indipendente dal kernel genitore.

=== Memoria in CUDA Dynamic Parallelism

#green_heading("Memoria Globale e Costante:")

- Le griglie parent e child *condividono lo stesso spazio di memoria globale* (accesso *concorrente*) *e memoria costante.* Tuttavia*, la memoria locale e condivisa* (shared memory) sono *distinte* fra parent e child*.*
- La coerenza della memoria globale non è garantita tra parent e child (be careful), tranne che:
  - *All'avvio della griglia child.*
  - *Quando la griglia child completa.*

#green_heading("Visibilità della Memoria:")

- Tutte le operazioni sulla memoria globale eseguite dal thread parent *prima* di lanciare una griglia child sono garantite essere *visibili e accessibili* ai thread della griglia child.
- Tutte le operazioni di memoria eseguite dalla griglia child sono garantite essere visibili al thread genitore *dopo che il genitore si è sincronizzato* con il completamento della griglia child.

#green_heading("Memoria Locale e Condivisa (Shared Memory):")

- La memoria locale e condivisa sono *private* per un thread o un blocco di thread, rispettivamente.
- La memoria locale e condivisa *non sono visibili* o *coerenti* tra parent e child.
- La memoria locale è uno spazio di archiviazione privato per un thread e *non è visibile al di fuori di quel thread*.

#green_heading("Limitazioni")

- *Non è valido* passare un puntatore a memoria locale o shared come argomento quando si lancia una griglia child.
- È possibile passare variabili *per copia* (by value).

#green_heading("Memoria in CUDA Dynamic Parallelism")

============ Memoria Globale e Costante:

#green_heading("Passaggio dei Puntatori alle Child Grid")

============ Possono Essere Passati

- Memoria Globale (sia variabili \_\_device\_
   sia memoria allocata con cudaMalloc)
- Memoria Zero-Host Copy
- Memoria Costante (ereditata dal parent e non può essere modificata)

========= Non Possono Essere Passati X

- Memoria Condivisa (variabili shared )
- Local Memory (incluse variabili dello stack)

\* Analizzeremo meglio queste memorie in seguito ("2.3 Modello di Memoria in CUDA")

============ Limitazioni

- Non è valido passare un puntatore a memoria locale o shared come argomento quando si lancia una griglia child.
- È possibile passare variabili per copia (by value).

=== Gestione dello Scambio Dati nel Parallelismo Dinamico

*Quindi, Come Restituire un Valore da un Child Kernel?*

```
__global__ void childKernel(void* p) { ... }
__global__ void parentKernel(void) {
 int v = 0; // Variabile nei registri/memoria locale del padre
 childKernel<<<16, 16>>>(&v); // Passa indirizzo non accessibile
 ...
}
                                 Versione Errata
```

#green_heading("Versione Corretta")

```
__device__ int v = 0; // Variabile in memoria globale
__global__ void childKernel(void* p) { ... }
__global__ void parentKernel(void) {
 childKernel<<<16, 16>>>(&v); // Passa indirizzo accessibile della memoria globale
 ...
}
```

=== Consistenza della Memoria nel Parallelismo Dinamico

#green_heading("Scenario Sicuro:")

- Quando il thread parent scrive in memoria globale *prima* di lanciare la griglia child.
- Il thread figlio vedrà *correttamente* il valore scritto dal padre.

#green_heading("Scenario Problematico:")

- *Scrittura da parte del child*:
  - Il thread parent potrebbe *non* vedere i valori scritti dal child.
- *Scrittura del parent dopo il lancio*:
  - Se il padre scrive dopo aver lanciato il figlio, si crea una "*race condition*".
  - Non si può sapere quale valore verrà letto.

```
__global__ void parentKernel(void) {
 v = 1; // OK
 childKernel<<<16,16>>>();
}
__device__ int v = 0; // Variabile globale
__global__ void childKernel( void ) {
 printf( "v = %d\n", v );
}
```

```
__global__ void parentKernel(void) {
 v = 1; // OK
 childKernel<<<16,16>>>();
 v = 2; // Race condition!
}
                                         Non c'è sincronizzazione esplicita
```

=== Dipendenze Annidate in CUDA

```
CPU
                                                      A
                                                      B
                                                     C
                                                                        X
                                                                        Y
                                                                        Z
                                                              GPU
__global__ void B(float *data)
{
 do_stuff(data);
 X <<< ... >>> (data);
 Y <<< ... >>> (data);
 Z <<< ... >>> (data);
 cudaDeviceSynchronize();
 do_more_stuff(data);
}
void main() 
{
 float *data;
 do_stuff(data);
 A <<< ... >>> (data);
 B <<< ... >>> (data);
 C <<< ... >>> (data);
 cudaDeviceSynchronize();
 do_more_stuff(data);
}
                 Stessa Sintassi
```

=== Sincronizzazione con cudaDeviceSynchronize()

#green_heading("Funzione Principale")

- *cudaDeviceSynchronize() attende il completamento* di tutte le griglie (kernel) precedentemente lanciate da qualsiasi thread del blocco corrente, includendo tutti i kernel discendenti (child, nipoti, ecc.) nella gerarchia.
- Se chiamata da un singolo thread, gli altri thread del blocco *continueranno* l'esecuzione.

#green_heading("Sincronizzazione a Livello di Blocco")

- *Attenzione*: *cudaDeviceSynchronize()* non implica una sincronizzazione fra thread del blocco.
- Il blocco di tutti i thread può essere ottenuto sia chiamando *cudaDeviceSynchronize()* da tutti i thread, sia facendo seguire la chiamata di *cudaDeviceSynchronize()* da parte di un singolo thread con *\_\_synchthreads()*.

```
__global__ void parentKernel(float *a, float *b, float *c) {
 createData(a, b); // Tutti i thread generano i dati
 __syncthreads(); // Sincronizzazione dei thread nel blocco per garantire i dati
 if (threadIdx.x == 0) {
 childKernel<<<n, m>>>(a, b, c); // Lancio della griglia child (1 thread call)
 cudaDeviceSynchronize(); // Attesa per il completamento dei kernel discendenti
 }
 __syncthreads(); // Tutti i thread nel blocco attendono prima di utilizzare i dati
 consumeData(c); // I thread nel blocco possono ora usare i dati della griglia child
}
```

#green_heading("Sincronizzazione con cudaDeviceSynchronize()")

========= Funzione Principale

cudaDeviceSynchronize() attende il completamento di tutte le griglie (kernel) precedentemente lanciate da qualsiasi thread del blocco corrente, includendo tutti i kernel discendenti (child, nipoti, ecc.) nella gerarchia.

============ Sincre

#green_heading("Limiti")

- cudaDeviceSynchronize() è un'operazione computazionalmente costosa perché:
  - Può causare la sospensione (swap-out) del blocco in esecuzione.
  - In caso di sospensione, richiede il trasferimento dell'intero stato del blocco (registri, memoria condivisa, program counter) nella memoria del device.
  - Il blocco dovrà poi essere ripristinato (swap-in) quando i kernel child saranno completati.
- Non dovrebbe essere chiamato <u>al termine</u> di un kernel genitore, poiché la *sincronizzazione implicita* viene già eseguita automaticamente.

\_\_syncthreads(); // Tutti i thread nel blocco attendono prima di utilizzare i dati
consumeData(c); // I thread nel blocco possono ora usare i dati della griglia child

cendo

Ĺ

=== Esecuzione Nidificata con CUDA Dynamic Parallelism

#image("images/_page_109_Figure_1_2.2.jpeg")

- *Esecuzione Nidificata*: Il thread CPU lancia la griglia parent (*blu*), che a sua volta lancia una griglia child (*verde*).
- *Sincronizzazione Esplicita*: La barriera nella griglia parent dimostra una *sincronizzazione esplicita* (*cudaDeviceSynchronize*) con la griglia child, assicurando che il parent attenda il completamento del child.
- *Completamento Gerarchico*: La griglia parent si considera *completata* solo dopo che la griglia child ha terminato.

=== Parallelismo Dinamico su GPU: Nested Hello World

- Il kernel seguente è un esempio di come utilizzare la *parallelizzazione dinamica* sulla GPU per eseguire un kernel ricorsivo.
- Il kernel viene invocato dalla applicazione *host* con una griglia di 8 thread in un singolo blocco. Il thread 0 di questo grid invoca un *nuovo grid* con la metà dei thread, e così via fino a quando non rimane solo un thread.

#image("images/_page_110_Picture_3_2.2.jpeg")

```
__global__ void nestedHelloWorld(int const iSize, int iDepth) {
 int tid = threadIdx.x;
 printf("Recursion=%d: Hello World from thread %d block %d\n", 
 iDepth, tid, blockIdx.x);
 // Condizione di terminazione: 
 // se c'è solo un thread, termina la ricorsione
 if (iSize == 1) return;
 // Calcola il numero di thread per 
 // il prossimo livello (dimezza)
 int nthreads = iSize >> 1;
 // Solo il thread 0 lancia ricorsivamente una nuova grid,
 // se ci sono ancora thread da lanciare
 if (tid == 0) {
 // Ricorsione
 nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);

 // Stampa la profondità di esecuzione nidificata
 printf("-------> nested execution depth: %d\n", iDepth);
 }}
```

=== Nested Hello World : Compilazione ed Esecuzione

Per compilare il codice abilitando il parallelismo dinamico:

```
$ nvcc -arch=sm_86 -rdc=true -lcudadevrt nested_hello_world.cu -o nested_hello_world
   --rdc=True: Abilita Relocatable Device Code, necessario per il parallelismo dinamico.
   -lcudadevrt: Collega la CUDA Device Runtime Library (spesso implicito con -rdc=true).
   -arch: Specifica l'architettura di destinazione della GPU (min. Kepler per il parallelismo dinamico. Ampere in questo caso).
```

Profiling con *Nsight Compute* (tuttavia, il tracciamento dei kernel CDP per le architetture GPU Volta e superiori non è supportato).

#green_heading("Output (Terminale)")

```
./nestedHelloWorld Configuration: grid 1 block 8
Recursion=0: Hello World from thread 0 block 0
Recursion=0: Hello World from thread 1 block 0
Recursion=0: Hello World from thread 2 block 0
Recursion=0: Hello World from thread 3 block 0
Recursion=0: Hello World from thread 4 block 0
Recursion=0: Hello World from thread 5 block 0
Recursion=0: Hello World from thread 6 block 0
Recursion=0: Hello World from thread 7 block 0
-------> nested execution depth: 1
                                                       Recursion=1: Hello World from thread 0 block 0
                                                       Recursion=1: Hello World from thread 1 block 0
                                                       Recursion=1: Hello World from thread 2 block 0
                                                       Recursion=1: Hello World from thread 3 block 0
                                                       -------> nested execution depth: 2
                                                       Recursion=2: Hello World from thread 0 block 0
                                                       Recursion=2: Hello World from thread 1 block 0
                                                       -------> nested execution depth: 3
                                                       Recursion=3: Hello World from thread 0 block 0
```

=== Nested Hello World : Compilazione ed Esecuzione

Ora, si provi a invocare la griglia parent con 2 blocchi invece di uno solo:

```
$ ./nestedHelloWorld 2
```

Perché l'ID dei blocchi per le griglie child è sempre 0 nei messaggi di output? (vedi codice precedente)

```
nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);
```

#image("images/_page_112_Figure_5_2.2.jpeg")

============ Output (Terminale)

```
./nestedHelloWorld Configuration: grid 1 block 8
Recursion=0: Hello World from thread 0 block 1
Recursion=0: Hello World from thread 1 block 1
Recursion=0: Hello World from thread 2 block 1
Recursion=0: Hello World from thread 3 block 1
Recursion=0: Hello World from thread 4 block 1
Recursion=0: Hello World from thread 5 block 1
Recursion=0: Hello World from thread 6 block 1
Recursion=0: Hello World from thread 7 block 1
Recursion=0: Hello World from thread 0 block 0
Recursion=0: Hello World from thread 1 block 0
Recursion=0: Hello World from thread 2 block 0
Recursion=0: Hello World from thread 3 block 0
Recursion=0: Hello World from thread 4 block 0
Recursion=0: Hello World from thread 5 block 0
Recursion=0: Hello World from thread 6 block 0
Recursion=0: Hello World from thread 7 block 0
-------> nested execution depth: 1
-------> nested execution depth: 1
Recursion=1: Hello World from thread 0 block 0
Recursion=1: Hello World from thread 1 block 0
Recursion=1: Hello World from thread 2 block 0
Recursion=1: Hello World from thread 3 block 0
Recursion=1: Hello World from thread 0 block 0
Recursion=1: Hello World from thread 1 block 0
Recursion=1: Hello World from thread 2 block 0
Recursion=1: Hello World from thread 3 block 0
-------> nested execution depth: 2
-------> nested execution depth: 2
Recursion=2: Hello World from thread 0 block 0
Recursion=2: Hello World from thread 1 block 0
Recursion=2: Hello World from thread 0 block 0
Recursion=2: Hello World from thread 1 block 0
-------> nested execution depth: 3
Recursion=3: Hello World from thread 0 block 0
-------> nested execution depth: 3
Recursion=3: Hello World from thread 0 block 0
```

=== Restrizioni sul Parallelismo Dinamico

#green_heading("Compatibilità dei Dispositivi")

Supportato solo da device con capacità di calcolo *≥ 3.5*.

#green_heading("Limitazioni di Lancio")

I kernel *non* possono essere lanciati su device *fisicamente separati*.

#green_heading("Profondità Massima di Nidificazione")

- Nesting depth imitata a *24 livelli*
- Nella pratica, limitata dalla memoria richiesta dal *runtime* del device.
- Runtime riserva *memoria aggiuntiva* per sincronizzazione griglia padre-figlio.

#green_heading("Deprecazione")

- L'uso di *cudaDeviceSynchronize* nel *codice device* è stato *deprecato* in CUDA 11.6 (la versione host-side rimane supportata). Rimosso per compute capability > 9.0.
- Per GPU con compute capability < 9.0 (es. Tesla T4 in Google Colab) e versione di CUDA ≥ 11.6 è possibile *forzare il supporto* usando il flag di compilazione *-D CUDA\_FORCE\_CDP1\_IF\_SUPPORTED*.

=== Riferimenti Bibliografici

#green_heading("Testi Generali")

- Cheng, J., Grossman, M., McKercher, T. (2014). *Professional CUDA C Programming*. Wrox Pr Inc. (1^ edizione)
- Kirk, D. B., Hwu, W. W. (2022). *Programming Massively Parallel Processors*. Morgan Kaufmann (4^ edizione)

#green_heading("NVIDIA Docs")

// - CUDA Programming:
//   - <http://docs.nvidia.com/cuda/cuda-c-programming-guide/>
// - CUDA C Best Practices Guide
//   - <http://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>
// - CUDA University Courses
//   - <https://developer.nvidia.com/educators/existing-courses=2>a.com/educators/existing-courses=2>