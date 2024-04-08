//Shortcut for centered figure with image
#let cfigure(img, wth) = figure(image(img, width: wth))

Screenshot di gdb:
#cfigure("2024-04-08-12-27-49.png",100%)

- Come prima proviamo a scatenare un segmentation fault
- Dopo pochi tentativi vediamo che per fare ciò basta dare come argomento al programma una stringa di 16 caratteri
- Per controllare che ciò che viene scritto dopo questa sequenza sovrascriva l'indirizzo di ritorno aggiungiamo alle 16 "A" la stringa "BBBB"; si nota che ora l'indirizzo di ritorno è `0x42424242`, come ci si aspettava
- Con 
  ```bash
  (gdb) info function
  ```
  vediamo che ci sono diverse funzioni segrete
  #cfigure("2024-04-08-12-34-34.png",80%)
- Per `secret_function_rreal`
  ```bash
  run $(perl -e 'print "A"x16,"\xc9\x61\x55\x56"')
  ```