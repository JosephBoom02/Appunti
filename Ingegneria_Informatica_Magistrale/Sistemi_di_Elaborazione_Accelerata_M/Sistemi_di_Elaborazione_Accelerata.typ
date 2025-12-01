#import "@preview/orionotes:0.1.0": orionotes

#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *



//Codly
#show: codly-init.with()


#codly(
  languages: (
    rust: (name: "Rust", icon: "🦀", color: rgb("#CE412B")),
    py: (name: "Python", icon: "🐍", color: rgb("#4CAF50")),
    sh: (name: "Shell", icon: "💲", color: rgb("#89E051")),
    cpp: (
      name: "",
      icon: box(
        fill: rgb("#22c55e"),
        inset: 0.3em,
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



#set text(9pt)
#set text(font: "Segoe UI")
#show raw.where(block: false): set text(size: 8pt)




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



