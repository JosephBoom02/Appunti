#let title = "Domande di Teoria Elettrotecnica"
#let author = "Bumma Giuseppe"

#set document(title: title, author: author)

//Code to have bigger fraction in inline math 
#let dfrac(x,y) = math.display(math.frac(x,y))

#set par(justify: true, leading: 0.9em)

//Shortcut for centered figure with image
#let cfigure(img, wth) = figure(image(img, width: wth))




= Teorema di Tellegen (01/06/2022)
Si consideri una rete elettrica con $l$ tensioni di lato ed $l$ correnti di lato
che soddisfino le #underline[leggi di Kirchhoﬀ]. Si ha che $underline(sum_(k=1)^(l) v_k i_k=0)$ Se $underline(v)$ e $underline(i)$ rappresentano le tensioni e le corrispondenti correnti di lato in uno stesso istante, si ha che il teorema di Tellegen si riduce al principio #underline[di conservazione] delle #underline[potenze istantanee]. È possibile esprimere la potenza #underline[erogata] dai bipoli attivi come $sum_(h=1)^M P_h$ dove M è il numero di componenti che rispettano la convenzione #underline[del generatore], e la potenza #underline[assorbita] dai bipoli passivi come $sum_(j=1)^N P_j$ dove N è il numero di componenti che rispettano la convenzione #underline[dell' utilizzatore]. In questo
caso il teorema di Tellegen afferma che la #underline[sommatoria] delle potenze
elettriche #underline[generate] dai bipoli attivi è pari a quella delle potenze
elettriche #underline[assorbite] dai bipoli passivi come descritto da $underline(sum_(h=1)^M P_h = sum_(j=1)^N P_j)$.



= Teorema del massimo trasferimento di potenza attiva su un bipolo (11/06/2022)

#cfigure("Teorema_potenza.png", 30%)

È data una sorgente di alimentazione sinusoidale (bipolo) e si vuole
determinare qual è il valore dell'impedenza $overline(Z) = R + j X$ di carico tale
da estrarre la massima potenza attiva dalla sorgente. La potenza attiva #underline[assorbita] dall'impedenza di carico $overline(Z)$ può essere espressa nella
forma $underline(P = R I^2)$. Si rappresenta la sorgente con un bipolo Thevenin.\
N.B. $(overline(V)_0, overline(Z)_0 = R_0 + j X_0)$. \
Il quadrato del valore efficacie della corrente che circola nell'impedenza vale $underline(I^2 = frac(V_0^2, (R+R_0)^2 + (plus.minus X plus.minus X_0)^2) )$.
La corrente, e quindi la potenza attiva, può essere dapprima massimizzata minimizzando la
reattanza complessiva, ovvero quando $underline(X = -X_0)$. La potenza attiva assorbita dall'impedenza risulta quindi $underline( P = frac(R V_0^2,(R+R_0)^2) )$. La massimizzazione complessiva può essere ottenuta applicando il teorema di trasferimento della massima potenza valido per una rete algebrica. Il valore della resistenza $R$ risulta quindi $underline( R = R_0 )$. Si ha pertanto che il valore dell'impedenza $overline(Z)$ tale da estrarre la massima potenza risulta $underline( overline(Z) = overline(Z)_0^convolve )$.


= Circuiti dinamici del secondo ordine (05/07/2022)

Sia dato un circuito dinamico del secondo ordine. Per determinare la soluzione associata all'equazione omogenea si introduce #underline[il polinomio caratteristico] dell'equazione #underline[differenziale] di #underline[secondo] grado. Si distinguono tre casi caratterizzati da valore positivo negativo o nullo del #underline[discriminante] $Delta = underline( alpha^2 - omega_0^2 )$ dove $alpha$ è il #underline[coefficiente di smorzamento] e $omega_0$ è #underline[la pulsazione di risonanza].\
Se $Delta > 0$ avremo due soluzioni #underline[reali distinte] e il circuito si dice #underline[sovrasmorzato]. Se $Delta < 0$ avremo due soluzioni #underline[complesse coniugate] ed il circuito si dice #underline[sottosmorzato]. Infine se $Delta = 0$ avremo due soluzioni #underline[reali coincidenti] ed il circuito si dice #underline[criticamente smorzato].\ 
Dato un circuito RLC serie $alpha$ è pari a $underline( frac(R,2L) )$ e $omega_0$ è uguale a $underline( frac(1, L C) )$.

= Il trasformatore (09/09/2022)
Il trasformatore è costituito da un nucleo di materiale #underline[ferromagnetico] su cui
sono avvolti #underline[due avvolgimenti]: il primario, costituito da $n_1$ spire ed il secondario, costituito da $n_2$ spire. Quando il primario è alimentato con una
tensione $v_1$ ("tensione primaria"), #underline[alternata], ai capi dell'avvolgimento secondario si manifesta una tensione $v_2$ ("tensione secondaria"), #underline[isofrequenziale] con la tensione primaria. La tensione $v_2$ è generata da una fem #underline[trasformatorica].\ 
Se il secondario è chiuso su di un carico elettrico, il primario #underline[eroga] la corrente $i_1$ ("corrente primaria"), ed il secondario #underline[assorbe] la corrente $i_2$ (corrente secondaria), entrambe le correnti sono alternate, #underline[isofrequenziali] con le tensioni.\
Mediante il trasformatore è quindi possibile trasferire potenza elettrica
dall'avvolgimento primario a quello secondario, senza fare ricorso ad alcun
collegamento #underline[elettrico] tra i due avvolgimenti; il trasferimento di potenza avviene invece attraverso #underline[il campo magnetico] che è presente principalmente nel nucleo del trasformatore e che è in grado di scambiare energia con entrambi i circuiti.\
Facendo riferimento ai versi positivi per le correnti e per i flussi mostrati nella
figura di sopra, il flusso totale concatenato con l'avvolgimento 1 $(phi_(c_1))$ ed il flusso totale concatenato con l'avvolgimento 2 $(phi_(c_2))$ risultano rispettivamente $phi_(c_1) = n_1 phi + phi_(d_1)$ e $phi_(c_2) = -n_2 phi + phi_(d_2)$ dove $phi$ è il #underline[flusso "principale"] mentre $phi_(d_1)$ e $phi_(d_2)$ e sono flussi “dispersi” concatenati rispettivamente con l'intero avvolgimento 1 e con l'intero avvolgimento 2.\
Tenendo in considerazione la #underline[caduta di tensione ohmica], sugli avvolgimenti si ha che la tensione ai capi del primario e quella ai capi del secondario sono rispettivamente pari a $underline( v_1(t) = frac( d phi_(c_1), d t ) + R_1 i_1 = n_1 frac( d phi, d t ) + frac( d phi_(d_1), d t ) + R_1 i_1 )$ e $underline( v_2(t) = - frac( d phi_(c_2), d t ) - R_2 i_2 = n_2 frac( d phi, d t ) - frac( d phi_(d_2), d t ) - R_2 i_2 )$.

= Rifasamento in monofase (22/07/2022)
Dato un sistema monofase alimentato da un generatore $e(t)$ e collegato ad un utilizzatore avente impedenza $overline(Z)_U$ (carico elettrico normalmente di tipo #underline[induttivo] con $overline(I)_L = overline(I)_U )$, la linea può essere
schematizzata tramite un'impedenza $underline( overline(Z)_L = R_L + j omega L )$. 
A causa della caduta di tensione su tale impedenza la tensione sul carico non è
uguale a quella generata ma varia in funzione del carico stesso.\
Alla resistenza di linea è associata una potenza elettrica dissipata per effetto joule pari a $underline( P_d = R_L I_L^2 )$.\ 
Applicando la #underline[legge di Kirchhoﬀ delle tensioni], la tensione applicata ai capi del carico risulta essere $underline( overline(V) = overline(E) - overline(Z)_L overline(I)_L )$. La potenza attiva assorbita dal carico viene definita come $underline(P = V I_L cos(phi))$, di conseguenza la corrente di linea viene espressa come $underline( I_L = frac(P, V cos(phi)) )$. Tale corrente può essere ridotta aumentando la tensione sul carico, riducendo la potenza attiva assorbita dal carico o #underline[aumentando] il $underline( cos(phi) )$, ovvero riducendo l'angolo di sfasamento tra tensione e corrente. Questo fa sì che corrente tensione relativi al carico siano maggiormente in #underline[fase].\
Per ridurre lo sfasamento è possibile introdurre un #underline[condensatore] in #underline[parallelo] al carico. La potenza #underline[reattiva] iniettata è di segno #underline[negativo], portando di conseguenza a diminuire la potenza #underline[apparente] del generatore. La corrente di linea risulta quindi pari a $underline( overline(I)'_L = overline(I)_U + overline(I)_C )$ di modulo #underline[inferiore] rispetto al caso privo di rifasamento.
