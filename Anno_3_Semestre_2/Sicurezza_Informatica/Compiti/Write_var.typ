//Shortcut for centered figure with image
#let cfigure(img, wth) = figure(image(img, width: wth))


Payload in `gdb`:
#cfigure("2024-04-08-11-33-51.png",100%)


Payload dato come argomento all'eseguibile:
#cfigure("2024-04-08-11-34-32.png",100%)

- Per poter riscrivere la variabile bisogna capire prima la lunghezza minima della stringa da dare come argomento tale per cui il programma da segmentation fault
- Quindi si inizia magari con una stringa di 100 "A", e poi a step di 10, 100 caratteri in più finché il programma non ci da segmentation fault
- Nel mio caso sono arrivato a dare una stringa di 2000 caratteri per scatenare il problema, e poi piano piano ho diminuito la lunghezza
- Ci viene in aiuto anche la stringa di debug che ci dice il valore della variabile `control`
- Infatti, dando come argomento una stringa lunga 1324 caratteri, vediamo che il valore di `control` è cambiato: se prima, con stringhe più corte, valeva `3039`, adesso con una stringa di 1324 caratteri ha valore `3030`; questo è il segnale che la sequenza di caratteri inserita dopo la sequenza di 1324 "A" verrà sovrascritta nella variabile `control`
- A questo punto dobbiamo costruire la sequenza di caratteri esadecimali con cui sovrascrivere la variabile, ricordando che il formato usato è little endian