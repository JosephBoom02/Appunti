#image("images/_page_0_Picture_0_2.2.jpeg")

=== Modello di Esecuzione CUDA

Sistemi di Elaborazione Accelerata, Modulo 2

A.A. 2025/2026

Fabio Tosi, Università di Bologna

=== Panoramica del Modello di Esecuzione CUDA

#green_heading("Architettura Hardware GPU")

- Introduzione al Modello di Esecuzione CUDA
- Organizzazione degli Streaming Multiprocessors (SM)
- Panoramica delle Architetture GPU NVIDIA

#green_heading("Organizzazione e Gestione dei Thread")

- Mappatura tra Vista Logica e Hardware
- Distribuzione e Schedulazione dei Blocchi sui SM

#green_heading("Modello di Esecuzione SIMT e Warp")

- Confronto tra SIMD e SIMT
- Warp e Gestione dei Warp
- Latency Hiding e Legge di Little
- Warp Divergence e Thread Independent Scheduling

#green_heading("Sincronizzazione e Comunicazione")

- Meccanismi di Sincronizzazione
- Operazioni Atomiche

#green_heading("Ottimizzazione delle Risorse")

- Resource Partitioning
- Occupancy

#green_heading("Parallelismo Avanzato")

CUDA Dynamic Parallelism

=== Introduzione al Modello di Esecuzione CUDA

#green_heading("Modello di Esecuzione CUDA")

In generale, un *modello di esecuzione* fornisce una *visione operativa* di come le istruzioni vengono eseguite su una specifica architettura di calcolo (nel nostro caso, le GPU).

#green_heading("Caratteristiche Principali")

- Fornisce un'*astrazione portabile dell'architettura* (Grid, Block, Thread, Warp, SM).
- Preserva *concetti fondamentali* tra generazioni differenti di GPU.
- Esposizione delle *funzionalità architetturali* chiave per la programmazione CUDA.
- Descrive come kernel, griglie e blocchi vengono effettivamente *mappati* sull'hardware GPU.
- Basato sul *parallelismo massivo* e sul *modello SIMT* (Single Instruction, Multiple Thread).

#green_heading("Importanza")

- Offre una *visione unificata* dell'esecuzione su diverse GPU.
- Fornisce indicazioni utili per *l'ottimizzazione* del codice in termini di:
  - *Throughput* delle istruzioni.
  - *Accessi alla memoria.*
- Facilita la comprensione della *relazione* tra il modello di programmazione e l'esecuzione effettiva.
- Permette di interpretare correttamente i risultati dei profiler CUDA, collegando i fenomeni osservati (latenze, occupancy, conflitti di memoria) alla struttura del modello di esecuzione.

=== Streaming Multiprocessor (SM)

#green_heading("Cosa sono?")

- Gli *Streaming Multiprocessors* (SM) sono le unità fondamentali di elaborazione all'interno delle GPU.
- Ogni SM contiene diverse *unità di calcolo*, *memoria condivisa* e *altre risorse essenziali* per gestire l'esecuzione concorrente e parallela di migliaia di thread.
- Il parallelismo hardware delle GPU è ottenuto attraverso la *replica* di questo blocco architetturale.

#image("images/_page_3_Figure_5_2.2.jpeg")

=== Streaming Multiprocessor (SM) Fermi SM (2010)

#green_heading("1. CUDA Cores")

Unità di elaborazione che eseguono istruzioni aritmetico/logiche.

#green_heading("2. Shared Memory/L1 Cache")

Memoria ad alta velocità condivisa tra i thread di un blocco.

#green_heading("3. Register Files")

Memoria privata di ogni thread per dati temporanei.

#green_heading("4. Load/Store Units (LD/ST)")

Gestiscono il trasferimento dati da/verso la memoria.

#green_heading("5. Special Function Units (SFU)")

Accelerano calcoli matematici complessi (funzioni trascendenti).

#green_heading("6. Warp Scheduler")

Seleziona thread pronti per l'esecuzione nell'SM.

#green_heading("7. Dispatch Unit")

Assegna i thread selezionati alle unità di esecuzione.

#green_heading("8. Instruction Cache")

Memorizza temporaneamente le istruzioni usate di frequente.

#image("images/_page_4_Figure_18_2.2.jpeg")

Uniform Cache 64 KB Shared Memory / L1 Cache

=== CUDA Core - Unità di Elaborazione CUDA

#green_heading("Cos'è un CUDA Core?")

- Un *CUDA Core* è l'*unità di elaborazione* di base all'interno di un SM di una GPU NVIDIA.
- L'architettura e la funzionalità dei CUDA Core sono evolute nel tempo, passando da unità generiche a unità specializzate.

#image("images/_page_5_Picture_4_2.2.jpeg")

Fermi SM (2010)

#green_heading("Composizione e Funzionamento (Architettura Fermi e Precedenti)")

- Inizialmente, i CUDA Core erano unità di calcolo relativamente semplici, in grado di eseguire sia operazioni intere (INT) che in virgola mobile (FP) in un ciclo di clock (fully pipelined, non simultaneamente).
  - *ALU (Arithmetic Logic Unit):* Ogni CUDA Core contiene un'unità logico-aritmetica che esegue operazioni matematiche di base come addizioni, sottrazioni, moltiplicazioni e operazioni logiche.
  - *FPU (Floating Point Unit)*: Include anche una FPU per gestire le operazioni in virgola mobile, supportando principalmente calcoli a precisione singola (FP32).
- I CUDA Core usano *registri condivisi* a livello di Streaming Multiprocessor per memorizzare temporaneamente dati durante l'esecuzione dei thread.

#green_heading("CUDA Core - Unità di Elaborazione CUDA")

============ Evoluzione dell'Architettura (Kepler e successive)

Dall'architettura Kepler, NVIDIA ha introdotto la specializzazione delle unità di calcolo all'interno di uno SM:

General

- Unità FP64: Dedicate alle operazioni in virgola mobile a doppia precisione.
- Unità FP32: Dedicate alle operazioni in virgola mobile a singola precisione.
- Unità INT: Dedicate alle operazioni intere.

Grafica

- Tensor Core TC (Architettura Volta e successive): Unità specializzate particolarmente ottimizzate per moltiplicazioni fra matrici in *precisione ridotta/mista* (FP32, FP16, TF32, INT8, etc.).
- Ray Tracing Core RT (Ampere e successive): Unità dedicate per l'accelerazione del *ray tracing*.
- Unità di Texture: Ottimizzate per gestire *texture* e *operazioni di filtraggio*.
- Unità di Rasterizzazione: Utilizzate per la *rasterizzazion*e delle immagini durante il rendering.

============ Ruolo nel Modello CUDA

Esecuzione Parallela: Ogni unità di elaborazione esegue un thread in <u>parallelo</u> con altri nel medesimo SM.

========= Differenze rispetto alle CPU

- Semplicità Architetturale: Le varie unità di gestione all'interno di un SM sono più semplici rispetto ai core delle CPU, senza unità di controllo complesse, permettendo una maggiore densità di unità specializzate.
- Specializzazione: Mentre le CPU sono general purpose, le GPU, attraverso i CUDA Core e le unità specializzate, offrono performance elevate anche per compiti specifici come l'Intelligenza Artificale ed il rendering grafico.

=== Streaming Multiprocessor - Evoluzione

#image("images/_page_7_Figure_1_2.2.jpeg")

- Architettura *unificata* per grafica e calcolo.
- *Concurrent Kernel Execution*: Esecuzione simultanea di più kernel.
- Supporto *FMA* e calcolo scientifico avanzato (FP64, ECC)

#green_heading("[Kepler GK100X SMX \(2012\)](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/NVIDIA-Kepler-GK110-GK210-Architecture-Whitepaper.pdf)")

#image("images/_page_7_Figure_6_2.2.jpeg")

- Nuovo design *SMX* con 192 CUDA Cores → più parallelismo per SM.
- *Dynamic Parallelism*: I kernel GPU possono lanciare altri kernel senza intervento della CPU.
- *Hyper-Q*: Fino a 32 connessioni simultanee CPU-GPU per ridurre le latenze.

#green_heading("[Maxwell SM \(2014\)](https://www.techpowerup.com/gpu-specs/docs/nvidia-gtx-980.pdf=:~:text=In%20many%20DX11%20applications%2C%20the%20GTX%20750,factor%20PCs%2C%20in%20addition%20to%20mainstream%20desktops.)")

Streaming Multiprocessor Sub-Partition (SMSP)

#image("images/_page_7_Figure_12_2.2.jpeg")

- *•* Maggiore *efficienza energetica* rispetto a Kepler.
- *Delta Color Compression* per ridurre la larghezza di banda.
- *Divisione degli SM in 4* per ottimizzare l'uso delle risorse.

=== Streaming Multiprocessor - Evoluzione

#green_heading("[Pascal SM \(2016\)](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf) [Volta SM \(2017\)](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)")

#image("images/_page_8_Figure_2_2.2.jpeg")

- *•* Introduzione di *NVLink* per connessioni ad alta velocità.
- *•* Supporto *HMB2* (High Bandwidth Memory).
- *Miglioramento* della tecnologia *GPU Boost 3.0* per l'overclocking dinamico.
- Supporto nativo per *FP16* (half precision)
- *16nm FinFET*: Processo produttivo con 2x efficienza energetica rispetto a Maxwell (28nm).

#image("images/_page_8_Figure_9_2.2.jpeg")

- *•* Introduzione dei *Tensor Core* per accelerare l'AI.
- *Independent Thread Scheduling* per un'esecuzione più flessibile dei thread.
- *Supporto FP32/INT32*  simultaneo e *NVLink 2.0*

#green_heading("[Turing \(2018\)](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf)")

#image("images/_page_8_Figure_14_2.2.jpeg")

- Introduzione dei *RT Cores*: Ray-tracing in tempo reale.
- *DLSS* (Deep Learning Super Sampling)
- *Variable Rate Shading*

=== Streaming Multiprocessor - Evoluzione

#image("images/_page_9_Figure_2_2.2.jpeg")

- *•* Aumento del throughput FP32 per i CUDA Cores.
- *•* Miglioramenti per *Tensor Cores* e *RT Cores*.
- *GPU Scheduler* ottimizzato per la gestione del carico di lavoro.

#green_heading("[Ampere \(2020\)](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf) [Ada Lovelace \(2022\)](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)")

#image("images/_page_9_Figure_7_2.2.jpeg")

- *•* Incremento delle prestazioni FP32 con *più CUDA Cores*.
- *•* Introduzione di *DLSS 3.0* con frame generation.
- *RT Cores migliorati* per ray tracing avanzato.

#green_heading("[Hopper \(2022\)](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c)")

#image("images/_page_9_Figure_12_2.2.jpeg")

- *Tensor Cores di quarta generazione* per AI.
- *Multi-Instance GPU (MIG).*
- *Thread Block Clusters*
- *FP8 Precision* per migliorare il throughput.

#green_heading("[Blackwell \(2025\)](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)")

#image("images/_page_9_Figure_18_2.2.jpeg")

- *Tensor Core di quinta generazione* per AI accelerata
- *Seconda generazione* di *Transformer Engine*
- *RT Cores di 4° generazione*
- Supporto per *precision FP4 e MXFP6/MXFP4*

#green_heading("Streaming Multiprocessor (SM) - Evoluzione")

- Aumento di SM e CUDA Core: Ogni generazione ha generalmente aumentato il numero di SM e CUDA Core.
- Miglioramento del Parallelismo: L'aumento delle unità di elaborazione permettono un <u>maggiore parallelismo</u>, migliorando le prestazioni complessive della GPU.
- Calcolo CUDA Core Totali: Totale CUDA Core = (SM per GPU) × (CUDA Core per SM)

| Architettura                 | SM per GPU | CUDA Cores<br>FP32 per SM | Totale CUDA<br>FP32 Cores |
|------------------------------|------------|---------------------------|---------------------------|
| Tesla (GTX 200 series)       | 30         | 8                         | 240                       |
| Fermi (GTX 400/500 series)   | 16         | 32                        | 512                       |
| Kepler (GTX 600/700 series)  | 15         | 192                       | 2880                      |
| Maxwell (GTX 900 series)     | 16         | 128                       | 2048                      |
| Pascal (GTX 10 series)       | 20         | 128                       | 2560                      |
| Volta (Tesla V100)           | 80         | 64                        | 5120                      |
| Turing (RTX 20 series)       | 72         | 64                        | 4608                      |
| Ampere (RTX 30 series)       | 84         | 128                       | 10752                     |
| Ada Lovelace (RTX 40 series) | 128        | 128                       | 16384                     |
| Hopper (GH series)           | 144        | 128                       | 18432                     |
| Blackwell (RTX 50 series)    | 170        | 128                       | 21760                     |

Nota: I valori mostrati sono tipici dei modelli di punta. Possono esserci variazioni tra i diversi modelli di una stessa serie.

#green_heading("Streaming Multiprocessor (SM) - Evoluzione")

- Aumento di SM e CUDA Core: Ogni generazione ha generalmente aumentato il numero di SM e CUDA Core.
- Miglioramento del Parallelismo: L'aumento delle unità di elaborazione permettono un maggiore parallelismo,
   migliorando le prestazioni complessive della GPU.
- Calcolo Throughput Teorico: Throughput FP32=Totale CUDA Cores FP32 × 2 × Frequenza di clock.

| Architettura                 | Totale CUDA<br>FP32 Cores | Frequenza di<br>Clock (Ghz) | FP32 Throughput (TFLOPS) |
|------------------------------|---------------------------|-----------------------------|--------------------------|
| Tesla (GTX 200 series)       | 240                       | 1,4                         | ~0,67                    |
| Fermi (GTX 400/500 series)   | 512                       | 1,4                         | ~1,44                    |
| Kepler (GTX 600/700 series)  | 2880                      | 0,9                         | ~5,18                    |
| Maxwell (GTX 900 series)     | 2048                      | 1,1                         | ~4,51                    |
| Pascal (GTX 10 series)       | 2560                      | 1,6                         | ~8,19                    |
| Volta (Tesla V100)           | 5120                      | 1,53                        | ~15,62                   |
| Turing (RTX 20 series)       | 4608                      | 1,65                        | ~15,23                   |
| Ampere (RTX 30 series)       | 10752                     | 1,8                         | ~38,78                   |
| Ada Lovelace (RTX 40 series) | 16384                     | 2,5                         | ~81,92                   |
| Hopper (GH series)           | 18432                     | 1,8                         | ~66,56                   |
| Blackwell (RTX 50 series)    | 21760                     | 2,4                         | ~104,75                  |

Nota: I valori mostrati sono tipici dei modelli di punta. Possono esserci variazioni tra i diversi modelli di una stessa serie.

=== Tensor Core: Acceleratori per l'Intelligenza Artificiale (Volta+)

#green_heading("Cosa sono i Tensor Core?")

- *Unità di elaborazione specializzata* per operazioni tensoriali (array multidimensionali).
- Progettata per accelerare calcoli di *AI* e *HPC*  (Riduzione dei tempi di training e inferenza).
- Presenti in GPU NVIDIA RTX da Volta (2017) in poi.

#green_heading("Caratteristiche")

- Esegue operazioni *matrice-matrice* (es. GEMM General Matrix Multiply) in *precisione mista*.
- Supporta formati *FP8*, *FP16*, *FP32*, *FP64*, *INT8*, *INT4*, *BF16* e nuovi formati come *TF32 (TensorFloat-32).*
- Offrono un significativo *speedup* nel calcolo senza compromettere l'accuratezza.

#green_heading("Evoluzione")

- *Miglioramenti*: Volta → ..→ .. → Hopper → Blackwell
- Integrazione con CUDA, cuDNN, TensorRT

#image("images/_page_12_Picture_12_2.2.jpeg")

=== Tensor Core: Acceleratori di Intelligenza Artificiale

- *Fused Multiply-Add (FMA):* Un'operazione che combina una moltiplicazione e un'addizione di *scalari* in un unico passo, eseguendo . Un CUDA core esegue 1 FMA per ciclo di clock in FP32.
- *Matrix Multiply-Accumulate (MMA):* Operazione che calcola il prodotto di due *matrici* e somma il risultato a una terza matrice, eseguendo
- Per matrici ( ), ( ) e ( ), l'operazione produce ( ) e richiede operazioni FMA, dove ogni elemento di necessita di moltiplicazioni-addizioni.

#image("images/_page_13_Figure_4_2.2.jpeg")

#green_heading("Esecuzione Parallela")

- Ogni Tensor Core esegue *64 operazioni FMA (4x4x4)* in un singolo ciclo di clock, grazie al parallelismo interno.
- Per operazioni su matrici più grandi, queste vengono *decomposte in sottomatrici 4x4*.
- Più operazioni 4x4 vengono eseguite *in parallelo su diversi Tensor Cores*.

=== Evoluzione dei NVIDIA Tensor Core

*Perdita di Precisione*: Si è dimostrato che ha un impatto

minimo sull'accuratezza finale dei modelli di deep learning.

Le generazioni più recenti di GPU hanno ampliato la flessibilità dei Tensor Cores, supportando *dimensioni di matrici più grandi e/o sparse* con un maggiore numero di formati numerici.

#image("images/_page_14_Figure_2_2.2.jpeg")

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

=== SM, Thread Blocks e Risorse

#green_heading("Parallelismo Hardware")

Più SM per GPU permettono l'*esecuzione simultanea* di migliaia di thread (anche da kernel differenti).

#green_heading("Distribuzione dei Thread Blocks")

- Quando un kernel viene lanciato, i blocchi di vengono *automaticamente e dinamicamente distribuiti dal GigaThread Engine* (scheduler globale) agli SM disponibili.
- Le variabili di identificazione e dimensione (*gridDim*, *blockIdx*, *blockDim*, e *threadIdx)* sono rese disponibili ad ogni thread e condivise nello stesso SM.
- Una volta assegnato a un SM, un blocco rimane vincolato a quell'SM *per tutta la durata dell'esecuzione*.

#green_heading("Gestione delle Risorse e Scheduling")

- *Più blocchi di thread* possono essere assegnati *allo stesso SM* contemporaneamente.
- Lo scheduling dei blocchi dipende dalla *disponibilità delle risorse* dell'SM (registri, memoria condivisa) e dai *limiti architetturali* di ciascun SM (max blocks, max threads, etc.).
- Tipicamente, la maggior parte delle grid contiene *molti più blocchi di quanti possano essere eseguiti* in parallelo sugli SM disponibili.
- Il *runtime system* mantiene quindi una coda di blocchi in attesa, assegnandone di nuovi agli SM non appena quelli precedenti terminano l'esecuzione.

=== Corrispondenza tra Vista Logica e Vista Hardware

#image("images/_page_17_Picture_1_2.2.jpeg")

=== Corrispondenza tra Vista Logica e Vista Hardware

#image("images/_page_18_Picture_1_2.2.jpeg")

#green_heading("Corrispondenza tra Vista Logica e Vista Hardware")

#image("images/_page_19_Figure_1_2.2.jpeg")

=== Distribuzione dei Blocchi su Streaming Multiprocessors

#image("images/_page_20_Picture_1_2.2.jpeg")

Supponiamo di dover realizzare un algoritmo parallelo che effettui il calcolo parallelo su un'immagine.

=== Distribuzione dei Blocchi su Streaming Multiprocessors

- Il Gigathread Engine *smista* i blocchi di thread agli SM in base alle risorse disponibili.
- CUDA non garantisce l'ordine di esecuzione e non è possibile scambiare dati tra i blocchi.
- Ogni blocco viene elaborato in modo *indipendente*.

#image("images/_page_21_Picture_4_2.2.jpeg")

#image("images/_page_21_Picture_5_2.2.jpeg")

#image("images/_page_21_Figure_6_2.2.jpeg")

*Capacità Massima Raggiunta*

=== Distribuzione dei Blocchi su Streaming Multiprocessors

Quando un blocco completa l'esecuzione e libera le risorse, un nuovo blocco viene schedulato al suo posto nell'SM, e questo processo continua fino a quando tutti i blocchi del grid non sono stati elaborati.

#image("images/_page_22_Picture_2_2.2.jpeg")

=== Concetto di Wave in CUDA

#green_heading("Cosa si intende per Wave?")

- Un "*Wave*" rappresenta l'insieme dei blocchi di thread che vengono eseguiti simultaneamente su tutti gli SM della GPU in un dato momento.
- La *Full Wave Capacity*, invece, rappresenta la *capacità teorica massima* della GPU, ossia il numero totale di blocchi che possono essere residenti simultaneamente su tutti gli SM.

```
Full Wave Capacity = (Numero di SM) * (Numero massimo di blocchi attivi per SM)
```

*Attenzione*: Questo numero massimo di blocchi *dipende dall'architettura GPU (Compute Capability)* e *dalle risorse richieste da ciascun blocco* (come registri, memoria condivisa), che influenzano *l'occupancy* (lo vedremo).

#green_heading("Full Wave vs Partial Wave")

- *Full Wave*: tutti gli SM sono occupati al massimo della loro capacità → utilizzo 100%
- *Partial Wave*: solo parte degli SM è occupata, oppure non tutti al massimo → utilizzo < 100%
- *Esempio*: GPU con 80 SM e fino a 4 blocchi attivi per SM → *Full Wave Capacity* = 80 × 4 = 320 blocchi simultanei.
  - ‣ Se il kernel lancia:
    - *320 blocchi* → esegue in *1 full wave* (320/320 = 100% utilizzo)
    - *100 blocchi* → esegue in *1 partial wave* (100/320 = ~31% utilizzo)
    - Cosa succede se il numero di blocchi è *superiore alla capacità massima*? (es. 500 blocchi)

=== Numero di Waves per un Kernel CUDA

#green_heading("Calcolo del Numero di Waves")

- Quando si lanciano più blocchi di quelli che la GPU può gestire simultaneamente, l'esecuzione avviene in più *ondate successive (waves)*:
- Il numero di blocchi attivi è una proprietà *statica*, determinata dall'architettura e dalle risorse richieste dal kernel.

```
Numero di waves = (Blocchi totali) / (Full wave capacity)
           ⌈ ⌉
```

#green_heading("Esempio")

Consideriamo una GPU con *8 SM* e un kernel con *12 blocchi totali* che consente *1 solo blocco attivo per SM*.

- *Full wave capacity =* 8 SM × 1 blocco/SM = 8 blocchi
- *Numero di waves* = ⌈12 / 8⌉ = 2 waves

========= Esecuzione:

- *Wave 1 (full wave):* 8 blocchi su 8 SM → *utilizzo 100%*
- *Wave 2 (partial wave):* 4 blocchi su 8 SM *→ utilizzo 50%*
- *Efficienza media di esecuzione:* (8 + 4) / (8 + 8) = 12/16 = 75%
- Il secondo wave è un esempio di *tail effect*.

#image("images/_page_24_Figure_14_2.2.jpeg")

=== Scalabilità in CUDA

#green_heading("Cosa si intende?")

- Per *scalabilità* in CUDA ci si riferisce alla capacità di un'applicazione di migliorare le prestazioni proporzionalmente all'*aumento delle risorse* hardware disponibili.
- *Più SM* disponibili = *Più blocchi eseguiti* contemporaneamente = *Maggiore Parallelismo*.
- *Nessuna modifica al codice* richiesta per sfruttare hardware più potente.

#image("images/_page_25_Figure_5_2.2.jpeg")

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

========= SIMD (Single Instruction, Multiple Data)

- È un *modello di esecuzione parallela* in cui una singola istruzione opera simultaneamente su più elementi di dato, utilizzando *unità vettoriali dedicate* (vector units) presenti nei core della CPU
- Utilizza *registri vettoriali* che possono contenere più elementi (es. 4 float, 8 int16, 16 byte).
- Il programma segue un *flusso di controllo centralizzato* (singolo thread di controllo).
- *Limitazioni*:
  - Larghezza vettoriale *fissa* nell'hardware (es. AVX-512 consente 512 bit), limitando gli elementi per istruzione.
  - Tutti gli elementi vettoriali in un vettore vengono elaborati in *lockstep* (perfettamente sincroni).
  - *La divergenza non è ammessa:* se occorrono percorsi condizionali (*if*-*else*), si impiegano *maschere vettoriali* che selezionano gli elementi su cui applicare l'operazione.

========= Somma di Due Array (SIMD con Neon intrinsics - ARM)

```
void array_sum(uint32_t *a, uint32_t *b, uint32_t *c, int n){ // I dati vengono suddivisi in vettori di
for(int i=0; i<n; i+=4) { // dimensione fissa e il loop elabora 
 //calcola c[i], c[i+1], c[i+2], c[i+3] // questi vettori utilizzando istruzioni
 uint32x4_t a4 = vld1q_u32(a+i); // intrinsics con nomenclatura specifica 
 uint32x4_t b4 = vld1q_u32(b+i); // dell'architettura.
 uint32x4_t c4 = vaddq_u32(a4,b4);
 vst1q_u32(c+i,c4);
 }}
```

=== Modello di Esecuzione: SIMT

========= SIMT (Single Instruction, Multiple Thread)

- *Modello ibrido* adottato in CUDA che combina parallelismo multi-thread con esecuzione SIMD-like.
- *Caratteristiche Chiave*:
  - A differenza del SIMD, non ha un controllo centralizzato delle istruzioni.
  - Ogni thread possiede un proprio *Program Counter (PC)*, *registri* e *stato* indipendenti (maggiore flessibilità).
  - *Supporta divergenza* del flusso di controllo (thread possono avere percorsi di esecuzione indipendenti).
  - Hardware gestisce *automaticamente* la divergenza (trasparente al programmatore).
- *Implementazione*
  - In CUDA, i thread sono organizzati in gruppi di 32 chiamati *warps* (unità minima di esecuzione in un SM).
  - I thread in un warp iniziano insieme allo *stesso indirizzo del programma (PC),* ma possono divergere.
  - Divergenza in un warp causa *esecuzione seriale dei percorsi diversi*, riducendo l'efficienza (da evitare).
  - La divergenza è *gestita automaticamente dall'hardware*, ma con un impatto negativo sulle prestazioni.

#green_heading("Somma di Due Array (SIMT)")

```
__global__ void array_sum(float *A, float *B, float *C, int N) {
 int idx = blockDim.x * blockIdx.x + threadIdx.x;
 if (idx < N) C[idx] = A[idx] + B[idx];
} Questa riga rappresenta l'essenza del SIMT
```

#green_heading("Modello di Esecuzione: SIMT")

#image("images/_page_29_Figure_1_2.2.jpeg")

=== Modello di Esecuzione: SIMD vs. SIMT

|                     | SIMD                                                       | SIMT                                                                 |
|---------------------|------------------------------------------------------------|----------------------------------------------------------------------|
| Unità di Esecuzione | Un singolo thread controlla<br>vettori di dimensione fissa | Molti thread leggeri raggruppati in<br>warp (32 thread)              |
| Registri            | Registri vettoriali condivisi<br>tra le unità di calcolo   | Set completo di registri per thread                                  |
| Flessibilità        | Bassa: Stessa operazione per<br>tutti gli elementi vettore | Alta: Ogni thread può eseguire<br>operazioni e percorsi indipendenti |
| Indipendenza        | Non applicabile,<br>controllo centralizzato                | Ogni thread mantiene il proprio<br>stato di esecuzione (vedi nota*)  |
| Branching           | Gestito esplicitamente con<br>maschere (no divergenza)     | Gestito via hardware con thread<br>masking automatico                |
| Scalabilità         | Limitata dalla larghezza vettoriale                        | Massiva (migliaia/milioni di thread)                                 |
| Sincronizzazione    | Intrinseca (lock-step automatico)                          | Esplicita (es.,syncthreads)                                          |
| Utilizzo Tipico     | Estensioni CPU (SSE, AVX, NEON)                            | GPU Computing (CUDA, OpenCL)                                         |
|                     |                                                            |                                                                      |

*<sup>\*</sup>Nota*: Con l'architettura Volta, NVIDIA ha introdotto l'Independent Thread Scheduling. *Pre-Volta*, un Program Counter (PC) era condiviso per warp; *post-Volta*, ogni thread ha il proprio PC, migliorando la gestione della divergenza e la flessibilità d'esecuzione (verrà affrontato nelle prossime slide).

=== Modello di Esecuzione Gerarchico di CUDA

============ \_\_global\_\_ void array\_sum(float \A, float \B, float \C, int N) { int idx === blockDim.x \ blockIdx.x + threadIdx.x; if (idx < N) C[idx] === A[idx] + B[idx]; } Livello di Programmazione int main(int argc, char \\argv){ // ... // Chiamata del kernel array\_sum<<<gridDim,blockDim>>>(args) ; }

#image("images/_page_31_Figure_2_2.2.jpeg")

=== Modello di Esecuzione Gerarchico di CUDA

#image("images/_page_32_Picture_1_2.2.jpeg")

=== Warp: L'Unità Fondamentale di Esecuzione nelle SM

========= Distribuzione dei Thread Block

Quando si lancia una griglia di thread block, questi vengono *distribuiti* tra i diversi SM disponibili.

========= Partizionamento in Warp

I thread di un thread block vengono suddivisi in *warp di 32 thread (con ID consecutivi).*

========= Esecuzione SIMT

I thread in un warp eseguono la *stessa istruzione* su *dati diversi*, con possibilità di *divergenza*.

========= Esecuzione Logica vs Fisica

Thread eseguiti in parallelo *logicamente*, ma non sempre fisicamente.

========= Scheduling Dinamico (Warp Scheduler)

L'SM gestisce *dinamicamente* l'esecuzione di un numero limitato di warp, switchando efficientemente tra di essi.

========= Sincronizzazione

Possibile all'interno di un thread block, ma non tra thread block diversi.

#image("images/_page_33_Figure_13_2.2.jpeg")

=== Organizzazione dei Thread e Warp

#green_heading("Thread Blocks e Warp")

- *Punto di Vista Logico:* Un blocco di thread è una collezione di thread organizzati in un layout 1D, 2D o 3D.
- *Punto di Vista Hardware:* Un blocco di thread è una collezione 1D di warp. I thread in un blocco sono organizzati in un layout 1D e ogni insieme di 32 thread consecutivi (con ID consecutivi) forma un warp.

#green_heading("Esempio 1D")

Un blocco 1D con 128 thread viene suddiviso in 4 warp, ognuno composto da 32 thread (*ID Consecutivi*).

```
Warp 0: thread 0, thread 1, thread 2, ... thread 31
Warp 1: thread 32, thread 33, thread 34, ... thread 63
Warp 2: thread 64, thread 65, thread 66, ... thread 95
Warp 3: thread 96, thread 97, thread 98, ... thread 127
                                                         threadIdx.x
```

#green_heading("Thread Block N (Caso 1D)")

#image("images/_page_34_Figure_8_2.2.jpeg")

=== Organizzazione dei Thread e Warp

#green_heading("Thread Blocks e Warp")

- *Punto di Vista Logico:* Un blocco di thread è una collezione di thread organizzati in un layout 1D, 2D o 3D.
- *Punto di Vista Hardware:* Un blocco di thread è una collezione 1D di warp. I thread in un blocco sono organizzati in un layout 1D e ogni insieme di 32 thread consecutivi (con ID consecutivi) forma un warp.

#green_heading("Mapping Multidimensionale (2D e 3D)")

- Il *programmatore* usa *threadIdx* e *blockDim* per identificare i thread nel layout logico.
- Il *runtime CUDA* si occupa automaticamente di linearizzare gli indici multidimensionali in ordine row-major, raggruppare i thread in warp, gestire il mapping hardware.
- L'ID di un thread in un blocco multidimensionale viene calcolato usando *threadIdx* e *blockDim*

```
○ Caso 2D: 
○ Caso 3D:
      threadIdx.z * blockDim.y * blockDim.x +/
           + threadIdx.y * blockDim.x + threadIdx.x
            threadIdx.y * blockDim.x + threadIdx.x
                                                          Ci pensa il runtime CUDA
```

- *Calcolo del Numero di Warp: ceil(*ThreadsPerBlock/warpSize*)*
- L'hardware alloca sempre un numero *discreto* di warp.

#green_heading("Organizzazione dei Thread e Warp")

========= Thread Blocks e Warp

- Punto di Vista Logico: Un blocco di thread è una collezione di thread organizzati in un layout 1D, 2D o 3D.
- Punto di Vista Hardware: Un blocco di thread è una <u>collezione 1D di warp</u>. I thread in un blocco sono organizzati in un layout 1D e ogni insieme di 32 thread consecutivi (con ID consecutivi) forma un warp.

========= Mapping Multidimensionale (Caso 2D)

Esempio 2D: Un thread block 2D con 40 thread in x e 2 in y (80 thread totali) richiederà 3 warp (96 thread hardware). L'ultimo semi-warp (16 thread) sarà inattivo, consumando comunque risorse.

#image("images/_page_36_Figure_6_2.2.jpeg")

#green_heading("Warp: L'Unità Fondamentale di Esecuzione nell'SM")

- Un warp viene assegnato a una sub-partition, solitamente in base al suo ID, dove rimane fino al completamento.
- Una sub-partition gestisce un "pool" di warp concorrenti di dimensione fissa (es., Turing 8 warp, Volta 16 warp).

#image("images/_page_37_Figure_3_2.2.jpeg")

=== Compute Capability (CC) - Limiti su Blocchi e Thread

- La *Compute Capability (CC)* di NVIDIA è un numero che identifica le *caratteristiche* e le *capacità* di una GPU NVIDIA in termini di funzionalità supportate e limiti hardware.
- È composta da *due numeri*: il numero principale indica la *generazione* dell'architettura, mentre il numero secondario indica *revisioni* e *miglioramenti* all'interno di quella generazione.

|                       |              |           |                        |                     | *Valori concorrenti per singolo SM |
|-----------------------|--------------|-----------|------------------------|---------------------|------------------------------------|
| Compute<br>Capability | Architettura | Warp Size | Max Blocchi<br>per SM* | Max Warp<br>per SM* | Max Threads<br>per SM*             |
| 1.x                   | Tesla        | 32        | 8                      | 24/32               | 768/1024                           |
| 2.x                   | Fermi        | 32        | 8                      | 48                  | 1536                               |
| 3.x                   | Kepler       | 32        | 16                     | 64                  | 2048                               |
| 5.x                   | Maxwell      | 32        | 32                     | 64                  | 2048                               |
| 6.x                   | Pascal       | 32        | 32                     | 64                  | 2048                               |
| 7.x                   | Volta/Turing | 32        | 16/32                  | 32/64               | 1024/2048                          |
| 8.x                   | Ampere/Ada   | 32        | 16/24                  | 48/64               | 1536/2048                          |
| 9.x                   | Hopper       | 32        | 32                     | 64                  | 2048                               |
| 10.x/12.x             | Blackwell    | 32        | 32                     | 64/48               | 2048/1536                          |

[https://en.wikipedia.org/wiki/CUDA=Version\\_features\\_and\\_specifications](https://en.wikipedia.org/wiki/CUDA=Version_features_and_specifications)

#green_heading("Warp: Contesto di Esecuzione")

Il contesto di *esecuzione locale* di un warp in un SM contiene:

*/olta-* : Per warp */olta+* : Per threa

- *Program Counter (PC)*: Indica l'indirizzo della prossima istruzione da eseguire.
- *Call Stack*: Struttura dati che memorizza le informazioni sulle chiamate di funzione, inclusi gli indirizzi di ritorno, gli argomenti, array e strutture dati più grandi.
- Registri: Memoria veloce e <u>privata</u> per ogni thread, utilizzata per memorizzare variabili e dati temporanei.
- Memoria Condivisa: Memoria veloce e <u>condivisa</u> tra i thread di un blocco utile per comunicare.
- Thread Mask: Indica quali thread del warp sono attivi o inattivi durante l'esecuzione di un'istruzione.
- Stato di Esecuzione: Informazioni sullo stato corrente del warp (es. in esecuzione/in stallo/eleggibile).
- Warp ID: Identificatore che consente di distinguere i warp e calcolare l'offset nel register file per ogni thread nel warp.
- L'SM mantiene on-chip il contesto di ogni warp <u>per tutta la</u> <u>sua durata</u>, quindi il cambio di contesto è senza costo.

============ Warp

#image("images/_page_39_Figure_12_2.2.jpeg")

thread 0..31

#image("images/_page_39_Figure_14_2.2.jpeg")

=== Parallelismo a Livello di Warp nell'SM

```
Codice CUDA
__global__ void array_sum(float *A, float *B, float *C, int N) {
 int i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i < N) C[i] = A[i] + B[i];}
MOV R1, c[0x0][0x28] // Carica il parametro N
S2R R6, SR_CTAID.X // Ottiene blockIdx.x
S2R R3, SR_TID.X // Ottiene threadIdx.x
IMAD R6, R6, c[0x0][0x0], R3 // Calcola i
ISETP.GE.AND P0, PT, R6, c[0x0][0x178], PT // Verifica se i < N
@P0 EXIT // Esce se i >= N
MOV R7, 0x4 // Dimensione di un float (4 byte)
ULDC.64 UR4, c[0x0][0x118] // Carica indirizzo base
IMAD.WIDE R4, R6, R7, c[0x0][0x168] // Calcola indirizzo di A[i]
IMAD.WIDE R2, R6, R7, c[0x0][0x160] // Calcola indirizzo di B[i]
LDG.E R4, [R4.64] // Carica A[i]
LDG.E R3, [R2.64] // Carica B[i]
IMAD.WIDE R6, R6, R7, c[0x0][0x170] // Calcola indirizzo di C[i]
FADD R9, R4, R3 // Esegue A[i] + B[i]
STG.E [R6.64], R9 // Salva il risultato in C[i]
EXIT // Termina il kernel
Codice SASS (Assembly)
                                                PC*
                                                       PC*
                                                  PC*
                                                         PC*
                       Compilazione (Codice comune a tutti i thread della grid)
```

I warp possono appartenere anche a blocchi differenti

*Warp 5 Instruction 4*

*Warp 9 Instruction 1* 

*Warp 11 Instruction 8*

*…*

*Warp N Instruction 15*

#image("images/_page_40_Picture_8_2.2.jpeg")

*SMSP (SM Sub-Partition)*

=== Classificazione dei Thread Block e Warp

#green_heading("Thread Block Attivo (Active Block)")

- Un thread block viene considerato *attivo* (o *residente*) quando gli vengono allocate risorse di calcolo di un SM come registri e memoria condivisa (non significa che tutti i suoi warp siano in esecuzione simultaneamente sulle unità).
- I warp contenuti in un thread block attivo sono chiamati *warp attivi.*
- Il numero di blocchi/warp attivi in ciascun istante è *limitato dalle risorse* dell'SM (compute capability).

#green_heading("Tipi di Warp Attivi")

- 1. *Warp Selezionato (Selected Warp)*
  - Un warp in esecuzione attiva su un'unità di elaborazione (FP32, INT32, Tensor Core, etc.).
- 2. *Warp in Stallo (Stalled Warp)*
  - Un warp in attesa di dati o risorse, impossibilitato a proseguire l'esecuzione.
  - Cause comuni: latenza di memoria, dipendenze da istruzioni, sincronizzazioni.
- 3. *Warp Eleggibile/Candidato (Eligible Warp)*
  - Un warp pronto (ma ancora non scelto) per l'esecuzione, con tutte le risorse necessarie disponibili.
  - Condizioni per l'eleggibilità:
    - *Disponibilità Risorse*: I thread del warp devono essere allocabili sulle unità di esecuzione disponibili.
    - *Prontezza Dati*: Gli argomenti dell'istruzione corrente devono essere pronti (es. dati dalla memoria).
    - *Nessuna Dipendenza Bloccante*: Risolte tutte le dipendenze con le istruzioni precedenti.

#green_heading("Classificazione degli Stati dei Thread")

========= Thread all'interno di un Warp

- Un warp contiene sempre 32 thread, ma non tutti potrebbero essere logicamente attivi.
- Lo stato di ogni thread è tracciato attraverso una thread mask o maschera di attività (un registro hardware a 32 bit).

============ Stati dei Thread

- 1. Thread Attivo (Active Thread)
  - Esegue l'istruzione corrente del warp.
  - o Contribuisce attivamente all'esecuzione *SIMT*.

#image("images/_page_42_Figure_8_2.2.jpeg")

============ 2. Thread Inattivo (Inactive Thread)

- o *Divergenza*: Ha seguito un percorso diverso nel warp per istruzioni di controllo flusso, come salti condizionali.
- *Terminazione*: Ha completato la sua esecuzione prima di altri thread nel warp.
- *Padding*: I thread di padding sono utilizzati in situazioni in cui il numero totale di thread nel blocco non è un multiplo di 32, per garantire che il warp sia completamente riempito (puro overhead).

============ Gestione degli Stati

- Gli stati sono *gestiti automaticamente dall'hardware* attraverso maschere di esecuzione.
- La transizione tra stati è dinamica durante l'esecuzione, quindi il numero di thread attivi può variare nel tempo.

=== Scheduling dei Warp

#green_heading("Introduzione al Warp Scheduler")

- Un'*unità hardware* presente in più copie all'interno di ogni SM, responsabile della *selezione* e *assegnazione* dei warp alle unità di calcolo CUDA.
- *Obiettivo*: Massimizzare l'utilizzo delle risorse di calcolo dell'SM, selezionando in modo efficiente i warp pronti e minimizzando i tempi di inattività.
- *Latency Hiding*: Contribuiscono a nascondere la latenza eseguendo warp alternativi quando altri sono in stallo, garantendo un utilizzo efficace delle risorse computazionali (prossime slide).

#green_heading("Funzionamento Generale")

- *Processo di Schedulazione*: I warp scheduler all'interno di un SM selezionano i warp eleggibili ad ogni ciclo di clock e li inviano alle dispatch unit, responsabili dell'assegnazione effettiva alle unità di esecuzione.
- *Gestione degli Stalli*: Se un warp è in stallo, il warp scheduler seleziona un altro warp eleggibile per l'esecuzione, garantendo consentendo l'esecuzione continua e l'uso ottimale delle risorse di calcolo.
- *Cambio di Contesto*: Il cambio di contesto tra warp è estremamente rapido (on-chip per tutta la durata del warp) grazie alla partizione delle risorse di calcolo e alla struttura hardware della GPU.

========= Limiti Architettonici

- Il numero di warp attivi su un SM è limitato dalle risorse di calcolo. (Esempio: 64 warp concorrenti su un SM Kepler).
- Il numero di warp selezionati ad ogni ciclo è limitato dal numero di scheduler di warp. (Esempio: 4 su un SM Kepler).

=== Warp Scheduler e Dispatch Unit

#green_heading("Warp Scheduler")

- È il "*cervello strategico*" che decide *quali* warp mandare in esecuzione.
- *Monitora* continuamente lo *stato dei warp* per identificare quelli eleggibili.
- Gestisce la *priorità* e l'*ordine* di esecuzione dei warp, cercando di minimizzare le latenze (latency hiding).

#green_heading("Dispatch Unit")

- È il "*braccio esecutivo*" che si occupa di *come* eseguire i warp selezionati.
- Si occupa di:
  - *Decodificare le istruzioni* del warp.
  - *Distribuire i thread* del warp alle unità di calcolo appropriate (es. FP, INT, Tensor Cores).
  - *Recuperare i dati* dai registri e dalla memoria necessaria per l'esecuzione.
  - *Assegnare fisicamente le risorse* hardware (registri, unità di calcolo) ai thread.

#green_heading("Fermi SM (2010)")

#image("images/_page_44_Figure_13_2.2.jpeg")

=== Scheduling dei Warp: TLP e ILP

#green_heading("Thread-Level Parallelism (TLP)")

- *Definizione*: Esecuzione simultanea di più warp per sfruttare il parallelismo tra thread.
- *Funzionamento*: Quando un warp è in attesa (ad esempio, per completare un'istruzione), un altro warp viene selezionato ed eseguito, aumentando l'occupazione delle unità di calcolo.

#green_heading("Instruction-Level Parallelism (ILP)")

- *Definizione*: Esecuzione di istruzioni indipendenti all'interno dello stesso warp.
- *Funzionamento:* Se ci sono più istruzioni pronte da eseguire in un warp, il warp scheduler può emettere queste istruzioni in parallelo alle unità di esecuzione, massimizzando l'utilizzo delle risorse (pipelining).

#green_heading("Importanza di TLP e ILP")

- *Massimizzazione delle Risorse*: TLP e ILP contribuiscono a mantenere le unità di calcolo attive e occupate, riducendo i tempi morti durante l'esecuzione.
- *Nascondere la Latenza*: TLP e ILP insieme aiutano a nascondere la latenza delle operazioni di memoria e di calcolo, migliorando le prestazioni complessive del sistema (vedi latency hiding).

=== Esecuzione Parallela dei Warp - Esempio con Fermi SM

#green_heading("Componenti Chiave per il Parallelismo")

- *Due Scheduler di Warp*: Selezionano due warp pronti da eseguire dai thread block assegnati all'SM.
- *Due Unità di Dispatch delle Istruzioni*: Inviano le istruzioni dei warp selezionati alle unità di esecuzione.

#green_heading("Flusso di Esecuzione")

- I blocchi vengono assegnati all'SM e *divisi in warp*.
- Due scheduler selezionano warp *pronti* per l'esecuzione.
- Ogni dispatch unit invia un'istruzione per warp a 16 CUDA Core, 16 unità di caricamento/memorizzazione (LD/ST), 4 unità di funzioni speciali (SFU).
- Questo processo *si ripete ciclicamente*, consentendo l'esecuzione parallela di più warp da più blocchi.

========= Capacità

Fermi (compute capability 2.x) può gestire simultaneamente *48 warp* per SM, per un totale di *1.536 thread residenti* in un singolo SM. Ad ogni ciclo, al più *2 selected warps*.

========= Fermi SM (2010)

#image("images/_page_46_Figure_12_2.2.jpeg")

Poiché le risorse di calcolo sono partizionate tra i warp e mantenute *on-chip* durante l'intero ciclo di vita del warp, il cambio di contesto tra warp è immediato.

#green_heading("Scheduling Dinamico dell Istruzioni - Fermi SM")

- Ad ogni ciclo di clock, un warp scheduler *emette un'istruzione* pronta per l'esecuzione.
- L'istruzione può provenire dallo stesso warp (ILP), se indipendente, o più spesso da un warp diverso (TLP).
- Se le risorse sono occupate, lo scheduler passa a un altro warp pronto (latency hiding).

#image("images/_page_47_Figure_4_2.2.jpeg")

#image("images/_page_47_Picture_5_2.2.jpeg")

=== Latency, Throughput e Concurrency

- *Mean Latency:* La latenza media è la *media delle latenze* degli elementi individuali. La latenza di un singolo elemento è la differenza tra il suo tempo di inizio e il suo tempo di fine.
- *Throughput:* Il throughput rappresenta la velocità di elaborazione. È definito come il *numero di elementi completati entro un dato intervallo di tempo* diviso per la durata dell'intervallo stesso.
- *Concurrency:* La concurrency misura *quanti elementi vengono processati contemporaneamente* in un determinato momento. Si può definire sia istantaneamente che come media su un intervallo di tempo.

#image("images/_page_48_Figure_4_2.2.jpeg")

=== Latency Hiding nelle GPU

#green_heading("Cosa è il Latency Hiding?")

- Una tecnica che permette di *mascherare i tempi di attesa* dovuti ad operazioni ad alta latenza (come gli accessi alla memoria globale) attraverso l'esecuzione concorrente di più warp all'interno di un SM.
- Si ottiene *intercambiando la computazione tra warp*, per massimizzare l'uso delle unità di calcolo di ogni SM.

#green_heading("Funzionamento")

- Ogni SM può gestire decine di warp concorrentemente da più blocchi (vedi compute capability della GPU).
- Quando un warp è in stallo (es. accesso memoria), l'SM passa immediatamente all'esecuzione di altri warp pronti.
- I Warp Scheduler dell'SM selezionano costantemente (ad ogni ciclo di clock) i warp pronti all'esecuzione (occorre che abbiano sempre warp eleggibili ad ogni ciclo).

#green_heading("Vantaggi del Latency Hiding")

- *Migliore Utilizzo delle Risorse*: Le unità di elaborazione della GPU sono mantenute costantemente attive.
- *Maggiore Throughput*: Completamento di un maggior numero di operazioni nello stessa unità di tempo.
- *Minore Latenza Effettiva*: Minimizza l'impatto delle operazioni ad alta latenza.

#green_heading("Tipi di Latenza (variano a seconda dell'architettura e dalla tipologia di operazione)")

- *Latenza Aritmetica*: Tempo di completamento di operazioni matematiche (bassa, es. 4-20 cicli).
- *Latenza di Memoria*: Tempo di accesso ai dati in memoria (alta, es. 400-800 cicli per la memoria globale).

=== Latency Hiding nelle GPU

#green_heading("Meccanismo dei Warp Scheduler")

- L'immagine mostra *due warp scheduler* che gestiscono l'esecuzione di diversi warp nel tempo.
- Warp Scheduler 0 e 1 *alternano l'esecuzione di warp diversi* per mantenere le unità di elaborazione occupate.
- Quando un warp è in attesa (es. Warp 0 all'inizio), *altri warp vengono eseguiti* per nascondere la latenza.
- I periodi di inattività (es. 'nessun eligible warp da eseguire') sono *minimizzati.*
- Questo approccio permette di *mascherare i tempi di latenza* e aumentare l'efficienza complessiva.
- Risorse pienamente utilizzate quando ogni scheduler ha un warp eleggibile ad *ogni ciclo di clock*.

#image("images/_page_50_Figure_8_2.2.jpeg")

=== Legge di Little

#green_heading("Cos'è la Legge di Little?")

La Legge di Little (dalla teoria delle code) ci aiuta a calcolare *quanti warp (approssimativamente) devono essere in esecuzione concorrente* per ottimizzare il latency hiding e mantenere le unità di elaborazione della GPU occupate.

#green_heading("Warp Richiesti = Latenza × Throughput")

- *Latenza:* Tempo di completamento di un'istruzione (in cicli di clock).
- *Throughput:* Numero di warp (e, quindi, di operazioni) eseguiti per ciclo di clock.
- *Warp Richiesti:* Numero di warp attivi necessari per nascondere la latenza.
- Indica che per nascondere la latenza, è necessario avere un *numero sufficiente di warp in esecuzione o pronti per l'esecuzione*, in modo che mentre uno è in attesa, altri possano essere eseguiti.

#green_heading("Note")

- La *latenza* e il *throughput* possono variare a seconda dell'architettura della GPU e del tipo di istruzioni.
- Questa è una *stima*, il numero effettivo di warp necessari potrebbe essere leggermente diverso.

=== Legge di Little

#green_heading("Esempio della Legge di Little")

*Latenza*: 5 Cicli

*Throughput desiderato*: 6 warp/ciclo

Numero di Warp Richiesti = 5 × 6 = 30 warp in-flight.

In questo caso, per mantenere un throughput di 6 warp/ciclo con una latenza di 5 cicli, avremmo bisogno di almeno 30 warp in esecuzione o pronti per l'esecuzione.

*Nota*: Un warp (32 thread) che esegue un'istruzione corrisponde a *32 operazioni*  (1 operazione per thread)

#image("images/_page_52_Picture_7_2.2.jpeg")

#green_heading("Massimizzare il Parallelismo per Operazioni Aritmetiche")

| Architettura | Latenza Istruzione<br>(Cicli) | Throughput<br>(Operazioni/Ciclo) | Parallelismo Necessario<br>(Operazioni) |
|--------------|-------------------------------|----------------------------------|-----------------------------------------|
| Fermi        | 20                            | 32 (1 warp/ciclo)                | 640 (20 warp)                           |
| Kepler       | 20                            | 192 (6 warp/ciclo)               | 3,840 (120 warp)                        |

Esempio: Operazione Multiply-Add a 32-bit Floating-Point (a + b  $\times$  c)

Limite Warp/SM in Kepler è 64

========= Consideriamo una GPU con architettura Fermi:

- Throughput: 32 operazioni/ciclo/SM
  - Un singolo SM può eseguire 32 operazioni di multiply-add a 32-bit floating-point per ciclo di clock.
- Warp Richiesti per SM: 640 ÷ 32 (operazioni per warp) = 20 warp/SM
  - Per raggiungere il throughput massimo e per mantenere il pieno utilizzo delle risorse computazionali, sono necessari 20 warp attivi contemporaneamente su ogni SM.

============ Esistono due modi principali per aumentare il parallelismo:

- ILP (Instruction-Level Parallelism): Aumentare il numero di istruzioni indipendenti all'interno di un singolo thread.
- TLP (Thread-Level Parallelism): Aumentare il numero di thread (e quindi di warp) che possono essere eseguiti contemporaneamente.

#green_heading("Massimizzare il Parallelismo per Operazioni di Memoria")

| Architettura | Latenza<br>(Cicli) | Bandwidth<br>(GB/s) | Bandwidth<br>(B/ciclo) | Parallelismo<br>(KB) |
|--------------|--------------------|---------------------|------------------------|----------------------|
| Fermi        | 800                | 144                 | 92                     | 74                   |
| Kepler       | 800                | 250                 | 96                     | 77                   |

========= Esempio: Operazione di Memoria

1/2

Consideriamo sempre una GPU con architettura Fermi:

- Calcolo del Bandwidth in Bytes/Ciclo:
  - o 144 GB/s ÷ 1.566 GHz ≈ 92 Bytes/Ciclo (<u>Frequenza di memoria</u> Fermi -Tesla C2070 = 1.566 GHz)
- Calcolo del Parallelismo Richiesto:
  - Parallelismo = Bandwidth (B/ciclo) × Latenza Memoria (cicli)
  - ∘ Fermi: 92 B/ciclo × 800 cicli ≈ 74 KB di I/O in-flight per saturare il bus di memoria.
- Memory Bandwidth è relativo all'intero device

- Interpretazione:
  - o 74 KB di operazioni di memoria necessarie per nascondere la latenza (per l'intero device, non per SM).

Recuperare la Memory Frequency di una GPU NVIDIA (da terminale)

```
$ nvidia-smi -a -q -d CLOCK | fgrep -A 3 "Max Clocks" | fgrep "Memory"
```

#green_heading("Massimizzare il Parallelismo per Operazioni di Memoria")

| Architettura | Latenza<br>(Cicli) | Bandwidth<br>(GB/s) | Bandwidth<br>(B/ciclo) | Parallelismo<br>(KB) |  |
|--------------|--------------------|---------------------|------------------------|----------------------|--|
| Fermi        | 800                | 144                 | 92                     | 74                   |  |
| Kepler       | 800                | 250                 | 96                     | 77                   |  |

============ Esempio: Operazione di Memoria

2/2

- Il legame tra questi valori e il numero di warp/thread varia a seconda della specifica applicazione.
- Conversione in Thread/Warp (Supponendo 4 bytes ad esempio, FP32 per thread):
  - $\circ$  74 KB ÷ 4 bytes/thread ≈ 18,500 thread
  - 18,500 thread  $\div$  32 thread/warp  $\approx$  579 warp
  - Per 16 SM: 579 warp ÷ 16 SM = 36 warp per SM
- Ovviamente, se ogni thread eseguisse più di un caricamento indipendente da 4 byte o un tipo di dato più grande (es. FP64), sarebbero necessari meno thread per mascherare la latenza di memoria.

============ Esistono due modi principali per aumentare il parallelismo di memoria:

- Maggiore Granularità: Spostare più dati per thread (ad esempio, caricare più float per thread).
- Più Thread Attivi: Aumentare il numero di thread concorrenti per aumentare il numero di warp attivi.

=== Warp Divergence

#green_heading("Cosa è la Warp Divergence?")

- In un warp, idealmente tutti i thread eseguono la *stessa istruzione contemporaneamente* per massimizzare il parallelismo SIMT (condividono un unico *Program Counter* [Architetture Pre-Volta] ).
- Tuttavia, se un'*istruzione condizionale* (come un *if*-*else* o *switch*) porta thread diversi a percorrere *rami diversi* del codice, si verifica la *Warp Divergence.*
- In questo caso, il warp esegue *serialmente ogni ramo*, utilizzando una *maschera di attività* (calcolata automaticamente in hardware) per abilitare/disabilitare i thread.
- La divergenza termina quando i thread *riconvergono* alla fine del costrutto condizionale.
- La Warp Divergence *può significativamente degradare le prestazioni* perché i thread non vengono eseguiti in parallelo durante la divergenza (le risorse non vengono pienamente utilizzate).
- Notare che il fenomeno della divergenza occorre *solo all'interno di un warp*.

#green_heading("Esempio")

```
if (threadIdx.x % 2 == 0) {
 // Istruzioni per thread con indice pari
} else {
 // Istruzioni per thread con indice dispari
}
```

=== CPU vs GPU: Gestione del Branching e della Warp Divergence

|                           | CPU                                                                       | GPU                                                                                                                         |
|---------------------------|---------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Esecuzione                | Singoli thread o piccoli gruppi,<br>indipendenti tra loro.                | Warp che eseguono le stesse<br>istruzioni concorrentemente.                                                                 |
| Branch Prediction         | Hardware dedicato, con algoritmi di<br>predizione complessi.              | Non supportata                                                                                                              |
| Esecuzione Speculativa    | Esegue istruzioni in anticipo<br>basandosi sulla branch prediction.       | Non supportata                                                                                                              |
| Impatto della Divergenza  | Mitigato dalla branch prediction e<br>dall'esecuzione speculativa.        | Causa la warp divergence, riducendo<br>il parallelismo e le prestazioni.                                                    |
| Gestione della Divergenza | Predizione del ramo più probabile e<br>esecuzione speculativa del codice. | Esecuzione seriale dei rami divergenti<br>nel warp, disabilitando i thread inattivi.                                        |
| Ottimizzazioni            | Meno critiche, gestite in parte<br>dall'hardware.                         | Branch predication (non lo vedremo) e<br>riorganizzazione del codice essenziali.                                            |
| Considerazioni            | Il costo della predizione errata<br>è relativamente basso.                | Il costo della warp divergence è elevato<br>a causa della perdita di parallelismo e<br>dell'overhead di esecuzione seriale. |

=== Warp Divergence: Analisi del Flusso di Esecuzione

#image("images/_page_58_Figure_1_2.2.jpeg")

#green_heading("Flusso")

- All'inizio, tutti i thread eseguono lo stesso codice (*blocchi blu*).
- Quando si incontra un'*istruzione condizionale* (*blocchi arancioni*), il warp si *divide*.
- Alcuni thread eseguono la clausola "*then*" (*blocchi verdi*), mentre altri sono in *stallo* (*blocchi viola*).
- Nei momenti di divergenza, l'efficienza può scendere al 50% (in questo caso, 16 thread attivi su 32).

=== Serializzazione nella Warp Divergence

#green_heading("Divergenza")

Quando i thread di un warp seguono percorsi diversi a causa di istruzioni condizionali (es. *if*), il warp esegue ogni ramo in serie, *disabilitando i thread inattivi*.

#green_heading("Località")

- La divergenza si verifica solo all'interno di un *singolo warp*.
- Warp diversi operano *indipendentemente*.
- I passi condizionali in *differenti warp* non causano divergenza.

#green_heading("Impatto")

La divergenza può ridurre il parallelismo *fino a 32 volte.*

#image("images/_page_59_Picture_9_2.2.jpeg")

=== Serializzazione nella Warp Divergence

#green_heading("Divergenza")

Quando i thread di un warp seguono percorsi diversi a causa di istruzioni condizionali (es. *if*), il warp esegue ogni ramo in serie, *disabilitando i thread inattivi*.

#green_heading("Località")

- La divergenza si verifica solo all'interno di un *singolo warp*.
- Warp diversi operano *indipendentemente*.
- I passi condizionali in *differenti warp* non causano divergenza.

#green_heading("Impatto")

La divergenza può ridurre il parallelismo *fino a 32 volte.*

#green_heading("Caso Peggiore")

```
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

=== Confronto delle Condizioni di Branch

*Kernel 1 Kernel 2*

```
__global__ void mathKernel1(float *sum) {
 int tid = blockIdx.x * blockDim.x + threadIdx.x;
 float a = 0.0f, b = 0.0f;
 if (tid % 2 == 0) a = 100.0f;
 else b = 200.0f;
 *sum += a + b;} // Race condition (risolveremo 
                  con atomicAdd dopo)
```

```
__global__ void mathKernel2(float *sum) {
 int tid = blockIdx.x * blockDim.x + threadIdx.x;
 float a = 0.0f, b = 0.0f;
 if ((tid / warpSize) % 2 == 0) a = 100.0f;
 else b = 200.0f;
 *sum += a + b;}
                   // Race condition (risolveremo 
                   con atomicAdd dopo)
```

============ Funzionamento

Valuta la parità dell'*ID* di ogni singolo thread.

============ Effetto sui thread

- *Thread pari* (ID 0, 2, 4, ...): eseguono il ramo *if*.
- *Thread dispari* (ID 1, 3,...): eseguono il ramo *else*.

============ Impatto sul warp

In ogni warp (32 thread), 16 thread eseguono *if* e 16 eseguono *else*.

============ Risultato

*Warp divergence*, con esecuzione serializzata dei due percorsi all'interno del warp.

============ Funzionamento

- tid/*warpSize*: Identifica l'*ID del warp* a cui appartiene il thread.
- (...) % 2: Determina la parità del numero del warp.

============ Effetto sui warp

- *Warp pari*: tutti i 32 thread eseguono il ramo *if*.
- *Warp dispari*: tutti i 32 thread eseguono il ramo *else*.

========= Impatto sul warp

Tutti i thread in un warp eseguono lo *stesso percorso.*

============ Risultato

*Eliminazione del warp divergence*, con esecuzione parallela all'interno di ogni warp (nessun overhead).

=== Confronto delle Condizioni di Branch

*Kernel 1 Kernel 2*

```
__global__ void mathKernel1(float *sum) {
 int tid = blockIdx.x * blockDim.x + threadIdx.x;
 float a = 0.0f, b = 0.0f;
 if (tid % 2 == 0) a = 100.0f;
 else b = 200.0f;
 *sum += a + b;} // Race condition (risolveremo 
                  con atomicAdd dopo)
```

```
__global__ void mathKernel2(float *sum) {
 int tid = blockIdx.x * blockDim.x + threadIdx.x;
 float a = 0.0f, b = 0.0f;
 if ((tid / warpSize) % 2 == 0) a = 100.0f;
 else b = 200.0f;
 *sum += a + b;}
                   // Race condition (risolveremo 
                   con atomicAdd dopo)
```

============ Branch Efficiency (calcolata in Nsight Compute)

La *Branch Efficiency* misura la percentuale di branch non divergenti rispetto al totale dei branch eseguiti da un warp.

Branch Efficiency = 
$$100 \times \left(\frac{\text{\= Branches} - \text{\= DivergentBranches}}{\text{\= Branches}}\right)$$

- Un *valore elevato* indica che la maggior parte dei branch eseguiti dal warp non causa divergenza.
- Un *valore basso* indica un'elevata divergenza, con conseguente perdita di prestazioni.

```
mathKernel1: Branch Efficiency 80.00%
mathKernel2: Branch Efficiency 100.00%
```

*Nota*: Nonostante la warp divergence, il compilatore CUDA applica ottimizzazioni anche con *-G* abilitato, risultando in una branch efficiency di *mathKernel1* superiore al 50% teorico.

=== Architetture Pre-Volta (< CC 7.0)

#green_heading("Pre-Volta 32 Thread Warp Program Counter (PC) and Stack (S)")

- *Singolo Program Counter* e *Call Stack* condiviso per tutti i 32 thread del warp (puntano alla stessa istruzione).
- Il warp agisce come una unità di esecuzione coesa/sincrona (stato dei thread è tracciato a livello di warp intero).
- *Maschera di Attività (Active Mask)* per specificare i thread attivi nel warp in ciascun istante.
- La maschera viene *salvata* fino alla riconvergenza del warp, poi *ripristinata* per riesecuzione sincrona.

#green_heading("Limitazioni")

- Quando c'è divergenza, i thread che prendono branch diverse *perdono concorrenza* fino alla riconvergenza.
- Possibili *deadlock* tra thread in un warp, se i thread dipendono l'uno dall'altro in modo circolare.

#green_heading("Esempio di Divergenza (Pseudo-Code) if (threadIdx.x < 4) { A; B; } else { X; Y; } Z;")

#image("images/_page_63_Figure_10_2.2.jpeg")

Una volta scelto un ramo, questo deve essere completato prima di poter iniziare l'altro (unico PC).

=== Architetture Pre-Volta (< CC 7.0)

#green_heading("Pre-Volta 32 Thread Warp Program Counter (PC) and Stack (S)")

- *Singolo Program Counter* e *Call Stack* condiviso per tutti i 32 thread del warp (puntano alla stessa istruzione).
- Il warp agisce come una unità di esecuzione coesa/sincrona (stato dei thread è tracciato a livello di warp intero).
- *Maschera di Attività (Active Mask)* per specificare i thread attivi nel warp.
- La maschera viene *salvata* fino alla riconvergenza del warp, poi *ripristinata* per riesecuzione sincrona.

#green_heading("Limitazioni")

- Quando c'è divergenza, i thread che prendono branch diverse *perdono concorrenza* fino alla riconvergenza.
- Possibili *deadlock* tra thread in un warp, se i thread dipendono l'uno dall'altro in modo circolare.

============ Esempio di Potenziale Deadlock diverge A; waitOnB(); ... Tempo if (threadIdx.x < 4) { A; waitOnB(); } else { B; waitOnA(); } Thread divergenti dello stesso warp non possono comunicare. B non verrà mai eseguito

#green_heading("Architettura Volta (CC 7.0+) e Independent Thread Scheduling")

============ Concetto chiave

L'Independent Thread Scheduling (ITS) consente <u>piena concorrenza tra i thread</u>, indipendentemente dal warp.

#image("images/_page_65_Picture_3_2.2.jpeg")

============ Stato di Esecuzione per Thread

- Ogni thread mantiene il *proprio stato di esecuzione*, inclusi program counter e stack di chiamate.
- Permette di <u>cedere l'esecuzione</u> a livello di *singolo thread* (non sono più obbligati a eseguire in lockstep).

============ Attesa per Dati

- Un thread può attendere che un altro thread produca dati, facilitando la comunicazione e la sincronizzazione tra di essi.
- Ottimizzazione della Pianificazione
  - Un ottimizzatore di scheduling raggruppa i thread attivi dello stesso warp in unità SIMT.
  - Così facendo, si mantiene l'alto throughput dell'esecuzione SIMT, come nelle GPU NVIDIA precedenti.

========= Flessibilità Maggiore

- I thread possono ora divergere e riconvergere indipendentemente con granularità sub-warp.
- Apre a pattern di programmazione che erano impossibili o problematici nelle architetture precedenti.

=== Confronto Pre-Volta vs Post-Volta

#image("images/_page_66_Figure_1_2.2.jpeg")

=== Confronto Pre-Volta vs Post-Volta

#image("images/_page_67_Figure_1_2.2.jpeg")

=== Introduzione di \_\_syncwarp in Volta

#green_heading("Scopo")

- Introdotta dall'architettura Volta *per supportare l'ITS* e *migliorare la gestione della divergenza* dei thread.
- Permette di *sincronizzare esplicitamente e riconvergere* i thread all'interno di un warp.
- *Blocca l'esecuzione* del thread corrente finché tutti i thread specificati nella maschera non hanno raggiunto il punto di sincronizzazione.

```
void __syncwarp(unsigned mask=0xffffffff);
```

#green_heading("Vantaggi")

- *Evita comportamenti non deterministici* dovuti alla divergenza intra-warp.
- Garantisce che tutti i thread del warp specificato *siano allineati prima di comunicare o accedere a dati condivisi*.
- Abilita l'*esecuzione sicura di algoritmi a grana fine* (riduzioni, scambi in shared memory, operazioni cooperative).

#green_heading("Esempio di Utilizzo")

```
if (threadIdx.x < 16) {
 // Codice per i primi 16 thread
} else {
 // Codice per gli ultimi 16 thread
}
```

```
Quando è davvero necessaria __syncwarp?
```

Dopo una divergenza:

È superflua se ogni thread lavora su dati privati, senza comunicare.

È necessaria se c'è comunicazione o dipendenza tra thread del warp.

*\_\_syncwarp*(); *// Sincronizza tutti i thread del warp qui*

=== Confronto Pre-Volta vs Post-Volta

|                            | Pre-Volta                                                 | Post-Volta                                                                 |
|----------------------------|-----------------------------------------------------------|----------------------------------------------------------------------------|
| Program Counter            | Singolo per warp                                          | Individuale per thread                                                     |
| Scheduling                 | Lockstep: tutti i thread del warp<br>eseguono insieme     | Indipendente: ogni thread può<br>progredire autonomamente                  |
| Sincronizzazione           | Implicita: i thread sono<br>automaticamente sincronizzati | Esplicita: richiedesyncwarp                                                |
| Divergenza                 | Serializzazione dei rami divergenti                       | Esecuzione interlacciata dei rami possibile                                |
| Deadlock Intra-Warp        | Possibili in certi scenari                                | Largamente mitigati                                                        |
| Prestazioni con Divergenza | Penalità per serializzazione                              | Penalità simile, nessun miglioramento<br>intrinseco                        |
| Complessità del Codice     | Workaround necessari per<br>certi algoritmi               | Implementazioni più naturali possibili<br>(ma richiede gestione esplicita) |

#green_heading("Confronto Pre-Volta vs Post-Volta")

Pre-Volta Post-Volta ITS: Limitazioni Prog ITS non può esonerare gli sviluppatori da una programmazione parallela impropria. Nessuno scheduling hardware può salvare dal livelock (ovvero thread che sono Sche tecnicamente in esecuzione ma non fanno progressi reali). Il progresso è garantito *solo per i warp residenti* al momento. I thread rimarranno in Sinc attesa infinita se il loro progresso dipende da un warp che non lo è. Non garantisce la riconvergenza, quindi le assunzioni relative alla programmazione a Dive ssibile livello di warp potrebbero non essere valide (usare esplicitamente syncwarp). Bisogna prestare più attenzione per garantire il comportamento SIMD dei warp. Dead ITS introduce *overhead hardware* per la gestione indipendente di program counter e Pres call stack per ogni thread, aumentando la flessibilità ma richiedendo più risorse. Implementazioni più naturali possibili Complessità del Codice Workaround necessari per certi algoritmi (ma richiede gestione esplicita)

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

=== Sincronizzazione in CUDA - Motivazioni

#green_heading("1. Asincronia tra Host e Device")

- *Comportamento di Base*: L'host e il device operano in modo *asincrono*.
- Senza sincronizzazione, l'host potrebbe tentare di utilizzare risultati *non ancora pronti* o *modificare dati ancora in uso* dalla GPU.

#green_heading("2. Sincronizzazione tra Thread all'Interno di un Blocco")

- *Comportamento di Base*: I thread all'interno di un blocco possono eseguire in ordine arbitrario e a velocità diverse.
- Quando i thread dello stesso blocco necessitano di condividere dati (utilizzando, ad esempio, la shared memory) o coordinare le loro azioni, è necessaria una sincronizzazione esplicita.

#green_heading("3. Coordinazione all'Interno dei Warp")

- *Comportamento di Base*:
  - *Pre-Volta*: I thread all'interno di un warp eseguivano sempre la stessa istruzione contemporaneamente (modello SIMD).
  - *Post-Volta* (CUDA 9.0+): Introdotta l'esecuzione indipendente dei thread (ITS) nel warp.
- Con l'esecuzione indipendente dei thread, la sincronizzazione esplicita diventa necessaria per garantire la *coerenza nelle operazioni intra-warp*.

=== Race Condition (Hazard)

#green_heading("Cos'è?")

Una *race condition* si verifica quando più thread accedono *concorrentemente (almeno uno in scrittura)* e in modo *non sincronizzato* alla stessa locazione di memoria, causando *risultati imprevedibili ed errori*.

#green_heading("Tipi di Race Condition (Noti anche nella pipeline dei processori)")

- *Read-After-Write (RAW):* Un thread legge prima che un altro finisca di scrivere.
- *Write-After-Read (WAR):* Un thread scrive dopo che un altro ha letto, invalidando il valore.
- *Write-After-Write (WAW):* Più thread scrivono nella stessa locazione, rendendo il valore indeterminato.

#green_heading("Perché si verificano?")

- I thread in un blocco sono logicamente paralleli ma non sempre fisicamente simultanei.
- Sono eseguiti in warp che possono trovarsi in punti diversi del codice.
- *Senza sincronizzazione*, l'ordine di esecuzione tra thread è *imprevedibile*.

#green_heading("Prevenzione delle Race Condition")

- *All'interno di un Thread Block*
  - *○* Utilizzare *\_\_syncthreads()* per sincronizzare i thread e garantire la visibilità dei dati condivisi.
  - *\_\_syncthreads()* garantisce che il thread A legga dopo che il thread B ha scritto.
- *Tra Thread Block diversi:*
  - Non esiste sincronizzazione diretta. L'unico modo sicuro è terminare il kernel e avviarne uno nuovo.

=== Deadlock in CUDA

#green_heading("Cos'è?")

- Un *deadlock* (o stallo) in CUDA si verifica quando i thread di un blocco si bloccano reciprocamente in attesa di sincronizzazioni o risorse non raggiungibili, causando il *blocco permanente* dell'esecuzione del kernel.
- Può insorgere in presenza di *sincronizzazioni condizionali* o *dipendenze* non gestite correttamente.

#green_heading("Condizioni per il Deadlock")

- *Sincronizzazione Condizionale:* Uso di *\_\_syncthreads()* all'interno di condizioni (*if*, *else*), dove solo una parte dei thread del blocco raggiunge il punto di sincronizzazione.
- *Dipendenze Circolari:* Situazioni in cui gruppi di thread attendono reciprocamente il completamento di operazioni, creando un ciclo di dipendenze irrisolvibile.
- *Risorse Condivise:* Gestione non corretta dell'accesso alla memoria condivisa o ad altre risorse comuni.

#green_heading("Prevenzione/Gestione del Deadlock")

- *Sincronizzazione Completa*: Evitare *\_\_syncthreads()* nei rami condizionali divergenti; assicurarsi che tutti i thread del blocco raggiungano i punti di sincronizzazione.
- *Ristrutturazione del Codice*: Rimuovere le dipendenze condizionali organizzando le operazioni in modo che tutti i thread completino una fase prima di passare alla successiva.
- *Independent Thread Scheduling*: Con architetture *Volta* e successive, i thread di un warp possono avanzare in modo più indipendente, grazie all'Independent Thread Scheduling ed alleviare il problema.

=== Sincronizzazione in CUDA

La sincronizzazione è il meccanismo che permette di *coordinare* l'esecuzione di thread paralleli e garantire la *correttezza* dei risultati, evitando *race condition*/*deadlock* e *accessi concorrenti non sicuri* alla memoria.

#green_heading("Livelli di Sincronizzazione in CUDA")

- *Livello di Sistema (Host-Device)*:
  - Blocca l'applicazione host finché tutte le operazioni sul device non sono completate.
  - Garantisce che il device abbia terminato l'esecuzione (copie, kernels, etc) prima che l'host proceda.
  - *○ Firma: cudaError\_t cudaDeviceSynchronize*(*void*); *// può causare overhead bloccando l'host*
- *Livello di Blocco (Thread Block)*:
  - Sincronizza tutti i thread all'interno di un singolo thread block.
  - Ogni thread attende che tutti gli altri thread nel blocco raggiungano il punto di sincronizzazione.
  - Garantisce la visibilità delle modifiche alla shared memory tra i thread del blocco.
  - *Firma: \_\_device\_\_ void \_\_syncthreads*(*void*); *// riduce le prestazioni se usato troppo*
- *Livello di Warp* (Disponibile con ITS a partire da CUDA 9.0 e architetture Volta+)
  - Sincronizza i thread all'interno di un singolo warp.
  - Garantisce la *riconvergenza* dei thread in presenza di divergenza.
  - *Ottimizza la cooperazione* tra thread dello stesso warp.
  - *Firma: \_\_device\_\_ void \_\_syncwarp*(*unsigned mask*=0xffffffff); *// minimo overhead*

=== Sincronizzazione in CUDA

La sincronizzazione è il meccanismo che permette di *coordinare* l'esecuzione di thread paralleli e garantire la *correttezza* dei risultati, evitando *race condition*/*deadlock* e *accessi concorrenti non sicuri* alla memoria.

#green_heading("Esempi")

#green_heading("Livello di Sistema")

```
__global__ void simpleKernel() {
 // Operazioni del kernel
}
int main() {
 ...
 simpleKernel<<<g, b>>>();
 cudaDeviceSynchronize();
 printf("Kernel completato\n");
 return 0;
}
```

```
__global__ void blockSyncKernel() {
 __shared__ int sharedData;
 if (threadIdx.x == 0) {
 sharedData = 42;
 }
 __syncthreads();
 if (threadIdx.x == 1) {
 printf("Valore condiviso:
%d\n",
 sharedData);
 }
}
```

========= Livello di Blocco Livello di Warp

```
__global__ void warpSyncKernel() 
{
 __shared__ int sharedData;
 if (threadIdx.x == 0) 
 sharedData = 99;
 __syncwarp();
 if (threadIdx.x < 32)
 printf("Thread %d, 
valore: %d\n", threadIdx.x, 
sharedData);
}
```

=== Operazioni Atomiche in CUDA

#green_heading("Perché sono Necessarie le Operazioni Atomiche?")

- *Problema:* Race condition in operazioni *Read-Modify-Write*
  - Più thread *accedono e modificano* la stessa locazione di memoria contemporaneamente.
  - Risultati *imprevedibili* e *inconsistenti*.

#green_heading("Scenario Tipico")

```
__global__ void increment(int *counter) {
 int old = *counter; // Legge il valore attuale dalla memoria
 old = old + 1; // Incrementa il valore letto
 *counter = old; // Scrive il nuovo valore nella stessa locazione
 }
```

#green_heading("Conseguenze")

- *Conteggi Errati:* Il valore finale potrebbe non riflettere correttamente il numero di incrementi eseguiti.
- *Aggiornamenti di Dati Persi:* Le modifiche apportate da alcuni thread potrebbero essere sovrascritte da altri.
- *Comportamento Non Deterministico:* L'applicazione potrebbe dare risultati diversi ad ogni esecuzione.

#green_heading("Soluzione")

Operazioni atomiche *garantiscono l'integrità* delle operazioni Read-Modify-Write in ambiente concorrente.

=== Operazioni Atomiche in CUDA

#green_heading("Cosa sono le Operazioni Atomiche? [\(Documentazione Online\)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html=atomic-functions)")

Operazioni *Read-Modify-Write* eseguite (solo su funzioni device) come *singola istruzione hardware indivisibile*.

#green_heading("Caratteristiche")

- *Esclusività dell'accesso alla memoria*: L'hardware assicura che solo un thread alla volta può eseguire l'operazione sulla stessa locazione di memoria. I thread che eseguono operazioni atomiche sulla stessa posizione vengono messi in coda ed eseguiti in serie (correttezza garantita).
- *Prevenzione delle interferenze*: Evitano che i thread interferiscano tra loro durante la modifica dei dati.
- *Compatibilità con memoria globale e condivisa*: Operano su word di 32, 64 bit o 128 bit.
- *Riduzione del parallelismo effettivo*, poiché i thread devono aspettare il proprio turno per accedere alla memoria.

#green_heading("Tipiche Operazioni Atomiche:")

- *Matematiche*: Addizione, sottrazione, massimo, minimo, incremento, decremento.
- *Bitwise*: Operazioni bit a bit come AND, OR, XOR.
- *Swap*: Scambio del valore in memoria con un nuovo valore.

#green_heading("Utilizzo di Base")

```
__global__ void safeIncrement(int *counter) {
 atomicAdd(counter, 1); // Incrementa il valore atomico, evitando condizioni di gara
}
```

=== Operazioni Atomiche in CUDA - Esempi d'Uso

========= Operazioni Atomiche

```
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

- Leggono il valore originale dalla memoria, eseguono l'operazione e salvano il risultato *nello stesso indirizzo*, *restituendo il valore originale pre-modifica*.
- *Supporto a Tipi Estesi:* Esistono anche varianti atomiche per operazioni su tipi a 64 bit (*long long int*) e floating point (*float* e *double*), supportate su architetture recenti.

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

=== Resource Partitioning in CUDA

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

=== Anatomia di un Thread Block

#green_heading("Requisiti di Risorse per SM")

Tutti i blocchi in una griglia eseguono lo stesso programma usando lo stesso numero di thread, portando a *3 requisiti di risorse fondamentali*:

#green_heading("1. Dimensione del Blocco")

Il numero di thread che devono essere concorrenti.

#green_heading("2. Memoria Condivisa")

È comune a tutti i thread dello stesso blocco.

#green_heading("3. Registri")

Dipendono dalla complessità del programma.

(*thread-per-blocco* × *registri-per-thread*)

============ Thread Block

#image("images/_page_82_Picture_11_2.2.jpeg")

Un blocco mantiene un numero costante di thread ed esegue unicamente su un singolo SM

```
__global__ void simpleKernel(float* out) {
 int tid = threadIdx.x;
 float myValue = 3.14;
 __shared__ float sharedData[64];
 sharedData[tid] = myValue;
 __syncthreads();
 out[tid] = sharedData[tid] * myValue;
}
```

*Risorse SM (Architettura Ampere)*

#image("images/_page_83_Figure_2_2.2.jpeg")

============ Thread Block

#image("images/_page_83_Picture_4_2.2.jpeg")

Un blocco contiene un numero fisso di thread ed esegue unicamente su un singolo SM

#green_heading("Esempio di Requisiti di Risorse per i Blocchi")

*Thread per Blocco* 256

*Registri per Thread* 64

*Registri per Blocco* (256\*64)= 16384

*Shared Memory per Blocco*

48Kb

*Risorse SM (Architettura Ampere)*

#image("images/_page_84_Figure_2_2.2.jpeg")

#image("images/_page_84_Picture_4_2.2.jpeg")

Un blocco contiene un numero fisso di thread ed esegue unicamente su un singolo SM

#green_heading("Esempio di Requisiti di Risorse per i Blocchi")

*Thread per Blocco* 256

*Registri per Thread* 64

*Registri per Blocco* (256\*64)= 16384

*Shared Memory per Blocco*

32Kb

#image("images/_page_85_Figure_1_2.2.jpeg")

*Risorse SM (Architettura Ampere)*

#image("images/_page_86_Figure_2_2.2.jpeg")

#green_heading("Esempio di Requisiti (Griglia Blu)")

*Thread per Blocco* 256

*Registri per Thread* 64

*Registri per Blocco* (256\*64)= 16384

*Shared Memory per Blocco* 48Kb

#green_heading("Esempio di Requisiti (Griglia Arancione)")

*Thread per Blocco* 512

*Registri per Thread* 32

*Registri per Blocco* (512\*32)= 16384

*Shared Memory per Blocco* 0

*Risorse SM (Architettura Ampere)*

#image("images/_page_87_Figure_2_2.2.jpeg")

#image("images/_page_87_Figure_3_2.2.jpeg")

#image("images/_page_87_Picture_4_2.2.jpeg")

Un blocco contiene un numero fisso di thread ed esegue unicamente su un singolo SM

#green_heading("Esempio di Requisiti di Risorse per i Blocchi")

*Thread per Blocco*

768

*Registri per Thread*

16

*Registri per Blocco*

(768\*16)= 12288

*Shared Memory per Blocco* 32 Kb

*Risorse SM (Architettura Ampere)*

#image("images/_page_88_Figure_2_2.2.jpeg")

============ Thread Block

#image("images/_page_88_Picture_4_2.2.jpeg")

Un blocco contiene un numero fisso di thread ed esegue unicamente su un singolo SM

#green_heading("Esempio di Requisiti di Risorse per i Blocchi")

*Thread per Blocco*

1024

*Registri per Thread*

16

*Registri per Blocco*

(1024\*16)= 16384

*Shared Memory per Blocco* 32 Kb

Anche se possiamo allocare solo 2 blocchi in entrambi i casi, perché la seconda configurazione è migliore?

#green_heading("Compute Capability (CC) - Limiti SM")

- La Compute Capability (CC) di NVIDIA è un numero che identifica le caratteristiche e le capacità di una GPU NVIDIA in termini di <u>funzionalità supportate</u> e <u>limiti hardware</u>.
- È composta da *due numeri*: il numero principale indica la *generazione* dell'architettura, mentre il numero secondario indica *revisioni* e *miglioramenti* all'interno di quella generazione.

| Compute<br>Capability | Architettura | Max Thread<br>per Blocco | Max Thread<br>per SM* | Max Warps<br>per SM* | Max Blocchi<br>per SM* | *Valori conce<br>Max Registri<br>per Thread | orrenti per singolo SM  Memoria Condivisa  per SM |
|-----------------------|--------------|--------------------------|-----------------------|----------------------|------------------------|---------------------------------------------|---------------------------------------------------|
| 1.x                   | Tesla        | 512                      | 768                   | 24/32                | 8                      | 124                                         | 16KB                                              |
| 2.x                   | Fermi        | 1024                     | 1536                  | 48                   | 8                      | 63                                          | 48KB                                              |
| 3.x                   | Kepler       | 1024                     | 2048                  | 64                   | 16                     | 255                                         | 48KB                                              |
| 5.x                   | Maxwell      | 1024                     | 2048                  | 64                   | 32                     | 255                                         | 64KB                                              |
| 6.x                   | Pascal       | 1024                     | 2048                  | 64                   | 32                     | 255                                         | 64KB                                              |
| 7.x                   | Volta/Turing | 1024 1                   | 024/2048              | 32/64                | 16/32                  | 255                                         | 96KB                                              |
| 8.x                   | Ampere/Ada   | 1024 1                   | 536/2048              | 48/64                | 16/24                  | 255                                         | 164KB                                             |
| 9.x                   | Hopper       | 1024                     | 2048                  | 64                   | 32                     | 255                                         | 228KB                                             |
| 10.x/12.x             | Blackwell    | 1024 2                   | 048/1536              | 64/48                | 32                     | 255                                         | 128KB                                             |

https://en.wikipedia.org/wiki/CUDA=Version\_features\_and\_specifications

=== Occupancy

#green_heading("Cosa è l'Occupancy?")

- L'occupancy rappresenta il *grado di utilizzo delle risorse* di calcolo dell'SM.
- L'occupancy è il *rapporto* tra i warp attivi e il numero massimo di warp supportati per SM (vedi compute capability):

*Occupancy [%]* = *Active Warps* / *Maximum Warps*

#green_heading("Punti Chiave")

- L'occupancy misura l'efficacia nell'uso delle risorse dell'SM:
  - *Occupancy Ottimale*: Quando raggiunge un livello sufficiente per nascondere la latenza. Un ulteriore aumento potrebbe degradare le prestazioni a causa della riduzione delle risorse disponibili per thread.
  - *Occupancy Bassa*: Risulta in una scarsa efficienza nell'emissione delle istruzioni, poiché non ci sono abbastanza warp eleggibili per nascondere la latenza tra istruzioni dipendenti.
- *Un'occupancy elevata non garantisce sempre prestazioni migliori*: Oltre certa soglia, fattori come i pattern di accesso alla memoria e il parallelismo delle istruzioni possono diventare più rilevanti per l'ottimizzazione.

#green_heading("Strumenti per l'Ottimizzazione")

- *Strumenti di Profiling:* Nsight Compute consente di recuperare facilmente l'occupancy, offrendo dettagli sul numero di warp attivi per SM e sull'efficienza delle risorse di calcolo (tuttavia, l'occupancy non deve mai essere guardata in isolamento. Diventa utile se combinata con altre metriche del profiler).
- *Suggerimento*: Osservare gli effetti sul tempo di esecuzione del kernel a diversi livelli di occupancy.

=== Occupancy Teorica vs Effettiva

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

=== Occupancy Teorica vs Effettiva

#green_heading("Obiettivi di Ottimizzazione")

- L'occupazione effettiva *non può* superare l'occupazione teorica (rappresenta il limite superiore).
- Pertanto, il primo passo per aumentare l'occupazione è *incrementare quella teorica*, modificando i fattori limitanti.
- Successivamente, è necessario verificare se il valore ottenuto è vicino a quello teorico per ridurre il gap.

#green_heading("Cause di Bassa Occupazione Effettiva")

- L'occupancy effettiva sarà inferiore a quella teorica quando il numero teorico di warp attivi non viene mantenuto durante l'attività dello SM (il problema forse non è il resource partitioning). Ciò può accadere quando si ha:
  - *Carico di lavoro sbilanciato nei blocchi*: Quando i warp in un blocco hanno tempi di esecuzione diversi (es. warp divergence), si crea un "*tail effect*" che riduce l'occupazione. Soluzione: bilanciare il carico tra i warp.
  - *Carico di lavoro sbilanciato tra blocchi*: Se i blocchi della grid hanno durate diverse, si può lanciare un maggior numero di blocchi o kernel concorrenti per ridurre l'effetto coda.
  - *Numero insufficiente di blocchi lanciati*: Se la grid ha meno blocchi del numero di SM del device, alcuni SM rimarranno inattivi. Ad esempio, lanciare 60 blocchi su un dispositivo con 80 SM lascia 20 SM sempre inattivi, riducendo l'utilizzo complessivo della GPU.
  - *Wave Parziale*: L'ultima ondata di blocchi potrebbe non saturare tutti gli SM. Ad esempio, con 80 SM che supportano 2 blocchi ciascuno e una grid di 250 blocchi: le prime due wave eseguono 160 blocchi (80 SM × 2), ma la terza wave ha solo 90 blocchi, lasciando alcuni SM parzialmente utilizzati o inattivi.

=== Nota Importante sull'Occupancy

#green_heading("Ricorda")

L'obiettivo finale *non è massimizzare l'occupancy*, ma *minimizzare il tempo di esecuzione del kernel.*

#green_heading("Linee guida pratiche")

- *Occupancy < 25–30 % → problema serio*
  - Non ci sono abbastanza warp per nascondere le latenze (critico soprattutto per kernel memory-bound).
  - La GPU risulta sottoutilizzata, con SM spesso inattivi
  - *Azione*: ridurre l'uso di registri / shared memory per thread oppure aumentare il numero di thread per blocco, se possibile.
- *Occupancy 50–80 % → generalmente buono*
  - Warp sufficienti per un buon *latency hiding* nella maggior parte dei casi
  - Risorse adeguate per ciascun thread
  - *Focus*: ottimizzare coalescing, divergenza, e accessi alla memoria
- *Occupancy > 90 % → non sempre vantaggiosa* (kernel memory bound compute bound )
  - Latency hiding ottimo ma attenzione: se limita troppo le risorse il guadagno si perde.
  - Oltre una certa soglia, più occupancy potrebbe non migliorare le performance.
- *Occupancy =* strumento per trovare il *giusto equilibrio tra latency hiding e risorse per thread*.

#green_heading("Nsight Compute: Occupancy Calculator")

Nsight Compute offre uno strumento utile chiamato "Occupancy Calculator" (<u>Documentazione</u>) che consente di:

- Stimare l'Occupancy: Calcola l'occupancy di un kernel CUDA su una determinata GPU.
- Ottimizzare le Risorse: Mostra l'impatto di registri e memoria condivisa sull'occupancy.
- *Migliorare le Prestazioni*: Fornisce suggerimenti per massimizzare l'uso delle risorse dell'SM e migliorare le prestazioni complessive.

#image("images/_page_94_Figure_5_2.2.jpeg")

#green_heading("Nsight Compute: Occupancy Calculator")

Nsight Compute offre uno strumento utile chiamato "Occupancy Calculator" (Documentazione) che consente di: Stimare l'Occupancy: Calcola l'occupancy di un kernel CUDA su una determinata GPU. Ottimizzare le Risorse: Mostra l'impatto di registri e memoria condivisa sull'occupancy. Linee Guida per le Dimensioni di Griglia e Blocchi Mantenere il numero di thread per block multiplo della dimensione del warp (32). Evitare dimensioni di block piccole: Iniziare con almeno 128 o 256 thread per block. Regolare la dimensione del blocco in base ai requisiti di risorse del kernel. Mantenere il *numero di blocchi molto maggiore del numero di SM* per esporre sufficiente parallelismo al dispositivo (latency hiding). Condurre esperimenti per scoprire la migliore configurazione di esecuzione e utilizzo delle risorse. Impatto della Variazione dell'Uso della Memoria Condivisa Per Blocco

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

- CUDA Programming:
  - <http://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- CUDA C Best Practices Guide
  - <http://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>
- CUDA University Courses
  - <https://developer.nvidia.com/educators/existing-courses=2>