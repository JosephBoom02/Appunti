# **Immagini come Matrici Multidimensionali**

## **Struttura di Base**

- Un'**immagine digitale** è una **griglia di pixel**.
- Ogni pixel rappresenta il **colore** o **l'intensità** di un punto specifico nell'immagine.
- Questa griglia può essere rappresentata matematicamente come una **matrice**.

## **Immagine a Colore (RGB) Immagine Grayscale**

![](_page_148_Picture_6.jpeg)

- **Dimensioni**: Larghezza x Altezza x 3 (canali)
- Ogni pixel è rappresentato da tre valori: Rosso, Verde, Blu (RGB).

![](_page_148_Figure_10.jpeg)

- **Dimensioni**: Larghezza x Altezza
- Ogni elemento della matrice è un singolo valore di intensità [0..255]

# **Immagini come Matrici Multidimensionali**

## **Struttura di Base**

- Un'**immagine digitale** è una **griglia di pixel**.
- Ogni pixel rappresenta il **colore** o **l'intensità** di un punto specifico nell'immagine.
- Questa griglia può essere rappresentata matematicamente come una **matrice**.

## **Immagine a Colore (RGB) Immagine Grayscale**

| <br>В <sub>00</sub> | В               |                 |                 |                 |                 |                 |                 |                 |
|---------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
|                     | G               | G.              | G.              | G               | G               | G               | G.              |                 |
| <br>B <sub>10</sub> |                 |                 | R <sub>01</sub> | R <sub>02</sub> | R <sub>03</sub> | R <sub>04</sub> | R <sub>05</sub> | R <sub>06</sub> |
| <br>В <sub>20</sub> | G <sub>10</sub> |                 | R <sub>11</sub> | R <sub>12</sub> | R <sub>13</sub> | R <sub>14</sub> | R <sub>15</sub> | R <sub>16</sub> |
| <br>В <sub>30</sub> |                 |                 | R <sub>21</sub> | R <sub>22</sub> | R <sub>23</sub> | R <sub>24</sub> | R <sub>25</sub> | R <sub>26</sub> |
| <br>B <sub>40</sub> |                 |                 | R <sub>31</sub> | R <sub>32</sub> | R <sub>33</sub> | R <sub>34</sub> | R <sub>35</sub> | R <sub>36</sub> |
| B <sub>50</sub>     | G <sub>40</sub> |                 | R <sub>41</sub> | R <sub>43</sub> | R <sub>43</sub> | R <sub>44</sub> | R <sub>45</sub> | R <sub>46</sub> |
|                     | - 50<br>-<br>-  | R <sub>50</sub> | R <sub>51</sub> | R <sub>53</sub> | R <sub>53</sub> | R <sub>54</sub> | R <sub>55</sub> | R <sub>56</sub> |

- **Dimensioni**: Larghezza x Altezza x 3 (canali)
- Ogni pixel è rappresentato da tre valori: Rosso, Verde, Blu (RGB).

| <br>I <sub>00</sub> | I <sub>01</sub> | I <sub>02</sub> | I <sub>03</sub> | I <sub>04</sub> | I <sub>05</sub> | Ι <sub>06</sub> |
|---------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| <br>I <sub>10</sub> | I <sub>11</sub> | I <sub>12</sub> | I <sub>13</sub> | I <sub>14</sub> | I <sub>15</sub> | I <sub>16</sub> |
| <br>I <sub>20</sub> | I <sub>21</sub> | I <sub>22</sub> | I <sub>23</sub> | I <sub>24</sub> | I <sub>25</sub> | I <sub>26</sub> |
| <br>I <sub>30</sub> | I <sub>31</sub> | I <sub>32</sub> | I <sub>33</sub> | I <sub>34</sub> | I <sub>35</sub> | I <sub>36</sub> |
| <br>I <sub>40</sub> | I <sub>41</sub> | I <sub>43</sub> | I <sub>43</sub> | I <sub>44</sub> | I <sub>45</sub> | I <sub>46</sub> |
| <br>I <sub>50</sub> | I <sub>51</sub> | I <sub>53</sub> | I <sub>53</sub> | I <sub>54</sub> | I <sub>55</sub> | I <sub>56</sub> |

- **Dimensioni**: Larghezza x Altezza
- Ogni elemento della matrice è un singolo valore di intensità [0..255]

# **Memorizzazione Lineare di Immagini RGB in CUDA**

- Per le immagini in **scala di grigi**, la memorizzazione in memoria globale è diretta e segue esattamente il principio **row-major** delle matrici classiche viste in precedenza.
- Per le immagini **RGB**, il principio di base rimane lo stesso, ma con una **complessità aggiuntiva** dovuta ai tre canali di colore (ogni pixel occupa 3 posizioni in memoria).

## **Approccio di Memorizzazione (Caso RGB)**

Ci sono due approcci principali per memorizzare un'immagine RGB in modo lineare:

- 1. **Planar**:
  - Tutti i valori R, poi tutti i G, poi tutti i B

![](_page_150_Figure_7.jpeg)

- 2. **Interleaved** (più comune):
  - I valori R, G, B per ogni pixel sono memorizzati consecutivamente

| R  | G  | B  | R  | G  | B  | R  | G  | B  | R  | G  | B  | <br> |  |
|----|----|----|----|----|----|----|----|----|----|----|----|------|--|
| 00 | 00 | 00 | 01 | 01 | 01 | 02 | 02 | 02 | 03 | 03 | 03 |      |  |

# **Accesso agli Elementi dell'Immagine**

![](_page_151_Picture_1.jpeg)

width (W)

**• Calcola l'indice di base**:

$$baseIndex = (i * width + j) * 3$$

- - **R**: baseIndex
  - **G**: baseIndex + 1
  - **B**: baseIndex + 2

![](_page_151_Figure_11.jpeg)

height (H)

Per accedere a un pixel specifico (i, j):

**• Accesso ai canali:**

$$baseIndex = (i * width + i) * 3$$

3

![](_page_151_Picture_28.jpeg)

![](_page_151_Picture_32.jpeg)

$$haseIndex = (i * width + i) * 3$$

![](_page_151_Figure_40.jpeg)

Per accedere a un pixel specifico (i, j):

baseIndex = i \* width + j

**RGB Grayscale** height (H)

![](_page_151_Figure_61.jpeg)

**• Calcola l'indice di base**:

width (W)

# **Parallelismo GPU nella Conversione RGB a Grayscale**

## **Perché le GPU sono Ideali per l'Elaborazione delle Immagini**

- **Struttura delle Immagini**
  - Le immagini sono composte da molti **pixel indipendenti.**
  - Ogni pixel può essere elaborato **separatamente.**
- **Operazioni Uniformi**
  - La **stessa operazione** viene spesso applicata a tutti i pixel.
  - Perfetto per il paradigma **SIMD** (Single Instruction, Multiple Data).

## **Esempio: Conversione RGB a Grayscale**

![](_page_152_Picture_9.jpeg)

![](_page_152_Picture_10.jpeg)

**Formula**: Gray = 0.299R + 0.587G + 0.114B (per pixel)

# **Suddivisione dell'Immagine in Blocchi per l'Elaborazione GPU**

- L'elaborazione di immagini su GPU richiede la **suddivisione** del lavoro in **unità parallele.**
- L'immagine viene divisa in una **griglia** di **blocchi**, ciascuno elaborato da un gruppo di thread.
  - **gridDim**: Numero di blocchi nella griglia.
  - **blockDim**: Numero di thread in ciascun blocco.

### **Calcolo degli indici nel buffer RGB**

```
ix = threadIdx.x + blockIdx.x * blockDim.x
iy = threadIdx.y + blockIdx.y * blockDim.y
base_index = (iy * width + ix) * 3
index_R = base_index
index_G = base_index + 1
index_B = base_index + 2
```

![](_page_153_Picture_8.jpeg)

### 

- Le dimensioni dei blocchi sono tipicamente potenze di 2.
- Le dimensioni delle immagini raramente sono multipli esatti di queste dimensioni dei blocchi.
- Per coprire l'intera immagine, si lanciano spesso **più blocchi del necessario**, alcuni dei quali si estendono **oltre i bordi** dell'immagine.
- I thread **che cadono fuori** dai limiti dell'immagine semplicemente **non eseguono** alcuna operazione.

```
ix = threadIdx.x + blockIdx.x * blockDim.x
iy = threadIdx.y + blockIdx.y * blockDim.y
base_index = (iy * width + ix) * 3
index_R = base_index
index_G = base_index + 1
index_B = base_index + 2
```

![](_page_154_Picture_8.jpeg)

# **Confronto: Conversione RGB a Grayscale in C vs CUDA C**

## **Codice C Standard**

```
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
// Funzione host per la conversione RGB->Gray
```

# **Confronto: Conversione RGB a Grayscale in C vs CUDA C**

## **Codice CUDA C**

```
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
// Funzione kernel per la conversione RGB->Gray
```

# **Conversione RGB a Grayscale in CUDA**

```
int main(int argc, char **argv) {
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
                                                                                        1/3
                               Conversione RGB -> Grayscale
```

## **Conversione RGB -> Grayscale**

```
 // Converti l'immagine in scala di grigi sulla CPU
 rgbToGrayscaleCPU(rgb, cpu_gray, width, height);
 // Alloca la memoria del device
 unsigned char *d_rgb, *d_gray;
 CHECK(cudaMalloc((void **)&d_rgb, rgbSize)); // Alloca memoria GPU per l'immagine RGB
 CHECK(cudaMalloc((void **)&d_gray, imageSize)); // Alloca memoria GPU per l'output 
 // Trasferisce i dati dall'host al device
 CHECK(cudaMemcpy(d_rgb, rgb, rgbSize, cudaMemcpyHostToDevice));
 // Configura e invoca il kernel CUDA
 dim3 block(32, 32); // Dimensione del blocco: 32x32 thread (altre dimensioni possibili)
 dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
 rgbToGrayscaleGPU<<<grid, block>>>(d_rgb, d_gray, width, height); // Lancia il kernel
 CHECK(cudaDeviceSynchronize()); // Aspetta il completamento del kernel
 // Copia il risultato del kernel dal device all'host
 CHECK(cudaMemcpy(h_gray, d_gray, imageSize, cudaMemcpyDeviceToHost));
```

**2/3**

## **Conversione RGB -> Grayscale**

```
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

**3/3**

- L'image flipping è una tecnica di elaborazione delle immagini che **inverte l'ordine dei pixel lungo un asse**
  - **Orizzontale:** Invertendo l'ordine dei pixel da sinistra a destra.

![](_page_160_Picture_6.jpeg)

![](_page_160_Picture_7.jpeg)

![](_page_160_Picture_8.jpeg)

## **Introduzione all'Image Flipping**

- specifico per ciascun canale di colore, creando un **effetto specchio**. Il flipping può essere:
  - **Verticale:** Invertendo l'ordine dei pixel dall'alto verso il basso.

height

![](_page_160_Picture_15.jpeg)

![](_page_160_Picture_16.jpeg)

![](_page_160_Picture_22.jpeg)

![](_page_160_Picture_24.jpeg)

![](_page_160_Picture_27.jpeg)

![](_page_160_Picture_30.jpeg)

![](_page_160_Picture_37.jpeg)

![](_page_160_Picture_43.jpeg)

![](_page_160_Picture_51.jpeg)

![](_page_160_Picture_52.jpeg)

- **•** In CUDA, ogni thread è responsabile del calcolo e della gestione di un singolo pixel dell'immagine.
  - Per un **flip orizzontale**, il thread calcola la nuova posizione speculare del pixel. Per un pixel inizialmente in posizione (i, j), il thread calcola la nuova posizione come (i, width -1 - j).
  - Per un **flip verticale**, la nuova posizione è calcolata come (height -1 i, j).
- Il thread **copia i valori** dei canali RGB del pixel originale nella nuova posizione calcolata.

![](_page_161_Picture_7.jpeg)

![](_page_161_Picture_8.jpeg)

![](_page_161_Picture_9.jpeg)

## **Processo di Flipping in CUDA**

![](_page_161_Picture_16.jpeg)

![](_page_161_Picture_17.jpeg)

![](_page_161_Picture_18.jpeg)

![](_page_161_Picture_25.jpeg)

![](_page_161_Picture_31.jpeg)

![](_page_161_Picture_39.jpeg)

**Input Image Flip Orizzontale Flip Verticale**

# **Image Flipping con CUDA**

## **Flipping di un'Immagine**

```
__global__ void cudaImageFlip(unsigned char* input, unsigned char* output,
 int width, int height, int channels, bool horizontal) {
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

## **Introduzione all'Image Blurring**

L'image blurring è una tecnica di elaborazione delle immagini che **riduce i dettagli** e le **variazioni di intensità**,

- **Riduzione del rumore**: Attenuando le fluttuazioni casuali dei pixel.
- **Enfasi degli oggetti**: Sfumando i dettagli irrilevanti e mettendo in risalto gli elementi principali.
- **Preprocessing per la Computer Vision**: Semplificando l'immagine per facilitarne l'analisi da parte degli algoritmi.

![](_page_163_Picture_6.jpeg)

creando un **effetto di sfocatura**. Viene utilizzata per:

![](_page_163_Picture_12.jpeg)

![](_page_163_Picture_14.jpeg)

**Input Image Blurred Image (**window\_size=25**)**

## **Concetto di Base**

Il blurring si ottiene calcolando la **media dei valori di intensità** dei pixel vicini di ogni pixel dell'immagine originale. L'operazione può essere riassunta come segue:

- **Pixel centrale:** Ogni pixel di output è la media dei pixel nella patch che lo circondano.
- **Esempio con patch 3×3:** Include il pixel centrale più gli 8 pixel che lo circondano, formando una matrice

![](_page_164_Figure_6.jpeg)

width

- **Patch di dimensioni N×N:** Una patch (o finestra) di dimensioni fisse scorre su ciascun pixel dell'immagine.
- di 3 righe e 3 colonne.

\*Se la precisione è importante, è possibile mantenere i valori come float o double.

## **Concetto di Base**

Il blurring si ottiene calcolando la **media dei valori di intensità** dei pixel vicini di ogni pixel dell'immagine originale. L'operazione può essere riassunta come segue:

- **Pixel centrale:** Ogni pixel di output è la media dei pixel nella patch che lo circondano.
- **Esempio con patch 3×3:** Include il pixel centrale più gli 8 pixel che lo circondano, formando una matrice

![](_page_165_Figure_6.jpeg)

width

- **Patch di dimensioni N×N:** Una patch (o finestra) di dimensioni fisse scorre su ciascun pixel dell'immagine.
- di 3 righe e 3 colonne.

### \*Se la precisione è importante, è possibile mantenere i valori come float o double.

## **Concetto di Base**

Il blurring si ottiene calcolando la **media dei valori di intensità** dei pixel vicini di ogni pixel dell'immagine originale. L'operazione può essere riassunta come segue:

- **Patch di dimensioni N×N:** Una patch (o finestra) di dimensioni fisse scorre su ciascun pixel dell'immagine.
- **Pixel centrale:** Ogni pixel di output è la media dei pixel nella patch che lo circondano.
- **Esempio con patch 3×3:** Include il pixel centrale più gli 8 pixel che lo circondano, formando una matrice di 3 righe e 3 colonne.

![](_page_166_Figure_7.jpeg)

## **Concetto di Base**

Il blurring si ottiene calcolando la **media dei valori di intensità** dei pixel vicini di ogni pixel dell'immagine originale. L'operazione può essere riassunta come segue:

- **Patch di dimensioni N×N:** Una patch (o finestra) di dimensioni fisse scorre su ciascun pixel dell'immagine.
- **Pixel centrale:** Ogni pixel di output è la media dei pixel nella patch che lo circondano.
- **Esempio con patch 3×3:** Include il pixel centrale più gli 8 pixel che lo circondano, formando una matrice di 3 righe e 3 colonne.

![](_page_167_Figure_7.jpeg)

## **Concetto di Base**

Il blurring si ottiene calcolando la **media dei valori di intensità** dei pixel vicini di ogni pixel dell'immagine originale. L'operazione può essere riassunta come segue:

- **Patch di dimensioni N×N:** Una patch (o finestra) di dimensioni fisse scorre su ciascun pixel dell'immagine.
- **Pixel centrale:** Ogni pixel di output è la media dei pixel nella patch che lo circondano.
- **Esempio con patch 3×3:** Include il pixel centrale più gli 8 pixel che lo circondano, formando una matrice di 3 righe e 3 colonne.

![](_page_168_Figure_7.jpeg)

## **Concetto di Base**

Il blurring si ottiene calcolando la **media dei valori di intensità** dei pixel vicini di ogni pixel dell'immagine originale. L'operazione può essere riassunta come segue:

- **Patch di dimensioni N×N:** Una patch (o finestra) di dimensioni fisse scorre su ciascun pixel dell'immagine.
- **Pixel centrale:** Ogni pixel di output è la media dei pixel nella patch che lo circondano.
- **Esempio con patch 3×3:** Include il pixel centrale più gli 8 pixel che lo circondano, formando una matrice di 3 righe e 3 colonne.

![](_page_169_Figure_6.jpeg)

## **Concetto di Base**

Il blurring si ottiene calcolando la **media dei valori di intensità** dei pixel vicini di ogni pixel dell'immagine originale. L'operazione può essere riassunta come segue:

- **Patch di dimensioni N×N:** Una patch (o finestra) di dimensioni fisse scorre su ciascun pixel dell'immagine.
- **Pixel centrale:** Ogni pixel di output è la media dei pixel nella patch che lo circondano.
- **Esempio con patch 3×3:** Include il pixel centrale più gli 8 pixel che lo circondano, formando una matrice di 3 righe e 3 colonne.

![](_page_170_Figure_6.jpeg)

\*Se la precisione è importante, è possibile mantenere i valori come float o double.

## **Concetto di Base**

Il blurring si ottiene calcolando la **media dei valori di intensità** dei pixel vicini di ogni pixel dell'immagine originale. L'operazione può essere riassunta come segue:

- **Patch di dimensioni N×N:** Una patch (o finestra) di dimensioni fisse scorre su ciascun pixel dell'immagine.
- **Pixel centrale:** Ogni pixel di output è la media dei pixel nella patch che lo circondano.
- **Esempio con patch 3×3:** Include il pixel centrale più gli 8 pixel che lo circondano, formando una matrice di 3 righe e 3 colonne.

![](_page_171_Figure_6.jpeg)

\*Se la precisione è importante, è possibile mantenere i valori come float o double.

Il blurring si ottiene calcolando la **media dei valori di intensità** dei pixel vicini di ogni pixel dell'immagine

- **Patch di dimensioni N×N:** Una patch (o finestra) di dimensioni fisse scorre su ciascun pixel dell'immagine.
- **Pixel centrale:** Ogni pixel di output è la media dei pixel nella patch che lo circondano.
- **Esempio con patch 3×3:** Include il pixel centrale più gli 8 pixel che lo circondano, formando una matrice di 3 righe e 3 colonne.

![](_page_172_Figure_5.jpeg)

## **Concetto di Base**

originale. L'operazione può essere riassunta come segue:

![](_page_172_Figure_12.jpeg)

## **Concetto di Base**

Il blurring si ottiene calcolando la **media dei valori di intensità** dei pixel vicini di ogni pixel dell'immagine originale. L'operazione può essere riassunta come segue:

- **Patch di dimensioni N×N:** Una patch (o finestra) di dimensioni fisse scorre su ciascun pixel dell'immagine.
- **Pixel centrale:** Ogni pixel di output è la media dei pixel nella patch che lo circondano.
- **Esempio con patch 3×3:** Include il pixel centrale più gli 8 pixel che lo circondano, formando una matrice di 3 righe e 3 colonne.

![](_page_173_Figure_6.jpeg)

\*Se la precisione è importante, è possibile mantenere i valori come float o double.

## **Caratteristiche Chiave del Kernel Blur**

- **Mappatura Thread-Pixel:** Ogni thread è responsabile del calcolo di un singolo pixel nell'immagine di output.
- **• Gestione dei Bordi:** Controlli specifici assicurano che la finestra di blur rimanga entro i confini dell'immagine,
- **• Parallelismo:** Il kernel sfrutta il parallelismo massiccio delle GPU, dato che il calcolo per ciascun pixel è
- **Pattern di Accesso alla Memoria**: Ogni thread accede a un vicinato di pixel (la patch) che, a seconda della disposizione dei dati in memoria, può comportare accessi **non sempre sequenziali**.

## **Confronto con Kernel Precedenti**

- **• Complessità:** Rispetto a semplici kernel come **vecAdd** (addizione vettoriale) o **rgbToGray** (conversione in scala di grigi), questo kernel è più complesso a causa della necessità di gestire più pixel e calcoli per ogni thread.
- **Accessi alla Memoria**: Ogni thread accede a più pixel rispetto a kernel semplici, aumentando la frequenza di accessi alla memoria globale.
- accessi alla memoria. Patch più grandi producono sfocature più intense ma richiedono più risorse.

- evitando letture di memoria non valide ai margini.
- indipendente dagli altri.

- **Scalabilità**: La dimensione della patch di blur (BLUR\_SIZE) impatta direttamente la quantità di calcolo e gli

## **Esercizio**

• Implementare in CUDA un kernel che applichi un filtro di blurring su un'immagine, sfruttando il parallelismo offerto dalle GPU per accelerare l'elaborazione rispetto a una soluzione sequenziale.

![](_page_175_Figure_3.jpeg)

# **Image Blur con CUDA: Soluzione (non ottimale)**

```
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

# **Introduzione alla Convoluzione 1D e 2D**

## **Che cos'è la Convoluzione?**

- Operazione matematica lineare **tra due funzioni**, segnale e kernel (fuorviante spesso indicato come **filtro**).
- Misura la **sovrapposizione** del filtro con il segnale mentre scorre su di esso.
- Produce una nuova funzione (segnale di output) che rappresenta le **caratteristiche estratte** dal segnale di input.

- **Convoluzione 1D**

- Il filtro è un vettore che **scorre** sul segnale.
- porzione di segnale sottostante.
- **Esempio**: Applicazione di un filtro di media mobile su un segnale audio per ridurre il rumore.

## **Convoluzione 2D**

- Applicata a **dati unidimensionali** (segnali audio, serie temporali, sequenze di testo).

- Applicata a **dati bidimensionali** (es. immagini).

- Il filtro è una matrice che **scorre** sull'immagine.

- L'output ad ogni punto è la **somma dei prodotti elemento per elemento (prodotto scalare)** tra il filtro e la

- - Applicazione di un filtro di **rilevamento dei bordi** a un'immagine per estrarre i contorni degli oggetti.

- L'output ad ogni pixel è la **somma dei prodotti elemento per elemento (prodotto scalare)** tra il filtro e la regione
- **Esempio**:

- - (Image Blur caso particolare di convoluzione 2D. Perchè?)
- dell'immagine sottostante.
  - Fondamentale nelle **reti neurali convoluzionali (CNN)** per l'elaborazione di immagini.

## Introduzione alla Convoluzione 1D e 2D

### 

- Operazione matematica lineare tra due funzioni, segnale e kernel (fuorviante spesso indicato come filtro).
- Misura la sovrapposizione del filtro con il segnale mentre scorre su di esso.

Proof pe socire poi poi e del pilo e con l'apegale e entre scorre su de esso

## Concetti Aggiuntivi

- por:

## 

- **Padding:** Consiste nell'aggiungere un **bordo di valori** (solitamente zeri) attorno all'input prima di applicare la convoluzione. Utile per controllare
- **Stride**: Definisce il passo con cui il kernel si sposta sull'input durante la convoluzione. Uno stride maggiore comporta sia una riduzione dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento del un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aumento dell'output che un aume

- App

- **Esempio**: (Image Blur caso particolare di convoluzione 2D. Perchè?)

Fondamentale nelle reti neurali convoluzionali (CNN) per l'elaborazione di immagini.

Applicazione di un filtro di rilevamento dei bordi a un'immagine per estrarre i contorni degli oggetti.

## 

### 

### 

$$O[i] = \sum_{j=-r}^{r} F[j], I[i+j]$$

![](_page_179_Figure_13.jpeg)

$$j=-r$$

![](_page_179_Figure_20.jpeg)

## 

### 

$$O[i] = \sum_{j=-r}^{r} F[j], I[i+j]$$

### 

$$O[i] = \sum_{j=-r}^{r} F[j], I[i+j]$$

![](_page_180_Figure_12.jpeg)

### 

- Input (I): Array di 7 elementi (I[0]...I[6]).
- Filtro (F): Array di 5 elementi (F[0]...F[4]).
- Output (O): Array risultante dalla convoluzione di l con F.

### 

$$O[i] = \sum_{j=-r}^{r} F[j], I[i+j]$$

![](_page_181_Figure_8.jpeg)

### 

- Filtro (F): Array di 5 elementi (F[0]...F[4]).
- Output (O): Array risultante dalla convoluzione di l con F.

### 

$$O[i] = \sum_{j=-r}^{r} F[j], I[i+j]$$

![](_page_182_Figure_8.jpeg)

## 

### 

$$O[i] = \sum_{j=-r}^{r} F[j], I[i+j]$$

![](_page_183_Figure_7.jpeg)

### 

# **Perché la Convoluzione si Adatta al Calcolo Parallelo**

## **Indipendenza dei Calcoli**

- Ogni elemento di output è calcolato **indipendentemente.**
- Permette l'**elaborazione parallela.**

## **Operazioni Uniformi**

- Stesse operazioni ripetute **su diverse porzioni dei dati.**
- Si allinea con l'architettura **SIMD.**

## **Mapping Diretto Thread-Output**

- **Ogni thread** può calcolare un elemento di output.
- Semplifica la parallelizzazione del problema.

## **Implementazione Generica: Passi**

- Un thread GPU **per ogni elemento** di output.
- Ogni thread:
  - **Identifica** regione input corrispondente.
  - **Applica** il filtro e **calcola** risultato.
  - **Scrive** output.

**Nota**: Questa è un'implementazione "naive". Ottimizzazioni avanzate saranno trattate successivamente.

# **CUDA Convoluzione 1D: Soluzione (non ottimale)**

```
int main() { // Parametri fittizi
 const int W = 1000, filterSize = 3; // Dimensioni input e filtro (dispari)
 float h_input[W]; // Input
 for (int i = 0; i < W; ++i) h_input[i] = static_cast<float>(i + 1); // Inizializzazione
 float h_filter[filterSize] = {0.2f, 0.5f, 0.2f}; // Filtro fittizio
 float h_output[W]; // Output per i risultati
 float *d_input, *d_output, *d_filter; // Puntatori su device 
 // Allocazione memoria su GPU
 cudaMalloc(&d_input, W * sizeof(float));
 cudaMalloc(&d_output, W * sizeof(float));
 cudaMalloc(&d_filter, filterSize * sizeof(float));
 // Copia input e filtro dalla CPU alla GPU
 cudaMemcpy(d_input, h_input, W * sizeof(float), cudaMemcpyHostToDevice);
 cudaMemcpy(d_filter, h_filter, filterSize * sizeof(float), cudaMemcpyHostToDevice);
 int blockSize = 256; // Numero di thread per blocco
 int numBlocks = (W + blockSize - 1) / blockSize; // Numero di blocchi
 // Lancio del kernel
 cudaConvolution1D <<<numBlocks, blockSize >>>(d_input, d_output, d_filter, W, filterSize);
 // continue..
                                                                                      1/2
```

# **CUDA Convoluzione 1D: Soluzione (non ottimale)**

```
__global__ void cudaConvolution1D(float* input, float* output, float* filter, int W, int
filterSize)
{
 int x = blockIdx.x * blockDim.x + threadIdx.x; // Indice globale del thread
 int radius = filterSize / 2; // Raggio del filtro (supponiamo filterSize dispari)
 if (x < W) // Verifica che il thread sia all'interno dei limiti dell'input
 {
 float result = 0.0f;
 for (int i = -radius; i <= radius; i++)
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
                                                                              2/2
```

### 

- Output (O): Matrice risultante dalla convoluzione di l

### 

$$O(x,y) = \sum_{m=-r_x}^{r_x} \sum_{n=-r_y}^{r_y} F[m,n], I[x+m,y+n]$$

![](_page_187_Figure_8.jpeg)

### 

- Input (I): Matrice di 25 elementi (I[0,0]...I[4,4]).
- Output (O): Matrice risultante dalla convoluzione di I con F.

### 

$$O(x,y) = \sum_{m=-r_x}^{r_x} \sum_{n=-r_y}^{r_y} F[m,n], I[x+m,y+n]$$

![](_page_188_Figure_8.jpeg)

### 

- Input (I): Matrice di 25 elementi (I[0,0]...I[4,4]).
- Output (O): Matrice risultante dalla convoluzione di l

 $O(x,y) = \sum_{m=-r_y}^{\infty} \sum_{n=-r_y}^{\infty} F[m,n], I[x+m,y+n]$ 

![](_page_189_Figure_7.jpeg)

### 

- Input (I): Matrice di 25 elementi (I[0,0]...I[4,4]).
- Output (O): Matrice risultante dalla convoluzione di l

### - Le dimente

$$O(x,y) = \sum_{m=-r_x}^{r_x} \sum_{n=-r_y}^{r_y} F[m,n], I[x+m,y+n]$$

![](_page_190_Figure_8.jpeg)

$$O(x,y) = \sum_{m=-r_x}^{r_x} \sum_{n=-r_y}^{r_y} F[m,n], I[x+m,y+n]$$

### 

- Output (O): Matrice risultante dalla convoluzione di l

### 

$$O(x,y) = \sum_{m=-r_x}^{r_x} \sum_{n=-r_y}^{r_y} F[m,n], I[x+m,y+n]$$

![](_page_191_Figure_15.jpeg)

$$O(x,y) = \sum_{m=-r_x}^{r_x} \sum_{n=-r_y}^{r_y} F[m,n], I[x+m,y+n]$$

### b-bocteto:pome-calcolor-convoluzione

Output (O): Matrice risultante dalla convoluzione di l

![](_page_192_Figure_11.jpeg)

### 

$$O(x,y) = \sum_{r_x}^{r_y} \sum_{r_y}^{r_y} F[m,n].I$$

$$O(x,y) = \sum_{m=-r_x}^{r_x} \sum_{n=-r_y}^{r_y} F[m,n], I[x+m,y+n]$$

![](_page_192_Figure_18.jpeg)

$$O(x,y) = \sum_{m=-r_x}^{r_x} \sum_{n=-r_y}^{r_y} F[m,n], I[x+m,y+n]$$

### 

- Output (O): Matrice risultante dalla convoluzione di l

### 

$$O(x,y) = \sum_{m=-r_x}^{r_x} \sum_{n=-r_y}^{r_y} F[m,n], I[x+m,y+n]$$

![](_page_193_Figure_14.jpeg)

### 

- Output (O): Matrice risultante dalla convoluzione di I con F.

### 

$$O(x,y) = \sum_{m=-r_x}^{r_x} \sum_{n=-r_y}^{r_y} F[m,n], I[x+m,y+n]$$

![](_page_194_Figure_8.jpeg)

### 

- Filtro (F): Matrice di 9 elementi (F[0,0]...F[2,2]).
- Output (O): Matrice risultante dalla convoluzione di I con F.

### 

$$O(x,y) = \sum_{m=-r_x}^{r_x} \sum_{n=-r_y}^{r_y} F[m,n], I[x+m,y+n]$$

 $r_{_{x}}$  , $r_{_{v}}$ : raggio del filtro 2D nelle due direzioni

![](_page_195_Figure_8.jpeg)

### 

- Output (O): Matrice risultante dalla convoluzione di I con F.

### 

$$O(x,y) = \sum_{m=-r_x}^{r_x} \sum_{n=-r_y}^{r_y} F[m,n], I[x+m,y+n]$$

![](_page_196_Figure_8.jpeg)

# **Convoluzione 2D: Rilevazione dei Contorni con Sobel**

- Il filtro di **Sobel** scorre sull'immagine pixel per pixel.
- Per ogni posizione:
  - Si **moltiplica** il filtro per la regione corrispondente dell'immagine.
  - Si **somma** i risultati per ottenere il valore del pixel di output.
- Evidenzia le **transizioni verticali** di intensità nell'immagine.

# **Filtro di Sobel**

| -1 | 0 | 1 |  |  |
|----|---|---|--|--|
| -2 | 0 | 2 |  |  |
| -1 | 0 | 1 |  |  |

![](_page_197_Picture_9.jpeg)

## **Verticale Processo di Convoluzione 2D**

![](_page_197_Picture_17.jpeg)

**Input Image Contorni Verticali**

![](_page_197_Figure_20.jpeg)

# **Convoluzione 2D: Rilevazione dei Contorni con Sobel**

- Per ogni posizione:
  - Si **moltiplica** il filtro per la regione corrispondente dell'immagine.
  - Si **somma** i risultati per ottenere il valore del pixel di output.
- Evidenzia le **transizioni orizzontali** di intensità nell'immagine.

# **Filtro di Sobel**

| -1 | -2 | -1 |  |  |
|----|----|----|--|--|
| 0  | 0  | 0  |  |  |
| -1 | -2 | -1 |  |  |

![](_page_198_Picture_7.jpeg)

## **Orizzontale Processo di Convoluzione 2D**

- Il filtro di **Sobel** scorre sull'immagine pixel per pixel

![](_page_198_Picture_15.jpeg)

![](_page_198_Picture_17.jpeg)

![](_page_198_Picture_18.jpeg)

**Input Image Contorni Orizzontali**

# **CUDA Convoluzione 2D: Soluzione (non ottimale)**

```
int main()
{
 // Parametri
 const int W, H, filterSize = 3;
 // Carica immagine usando stb_image.h
 // ..
 const int size = W * H * sizeof(float);
 const int filterBytes = filterSize * filterSize * sizeof(float);
 // Allocazione memoria host
 float* h_input = (float*)malloc(size);
 float* h_output = (float*)malloc(size);
 float* h_filter = (float*)malloc(filterBytes);
 // Inizializzazione filtro Sobel verticale (come esempio)
 float sobelY[9] = {-1.0f, -2.0f, -1.0f,
 0.0f, 0.0f, 0.0f,
 1.0f, 2.0f, 1.0f};
 memcpy(h_filter, sobelY, filterBytes);
                                                                                   1/3
```

![](_page_200_Picture_0.jpeg)

![](_page_201_Picture_0.jpeg)

![](_page_202_Picture_0.jpeg)

![](_page_203_Picture_0.jpeg)

# **Riferimenti Bibliografici**

## **Testi Generali**

- Cheng, J., Grossman, M., McKercher, T. (2014). **Professional CUDA C Programming**. Wrox Pr Inc. (1^ edizione)
- Kirk, D. B., Hwu, W. W. (2022). **Programming Massively Parallel Processors**. Morgan Kaufmann (4^ edizione)

## **NVIDIA Docs**

- CUDA Programming:
  - <http://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- CUDA C Best Practices Guide
  - <http://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>
- CUDA University Courses
  - <https://developer.nvidia.com/educators/existing-courses#2>