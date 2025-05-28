#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *


#let title = "Laboratorio di Sicurezza dell'Informazione M"
#let author = "Bumma Giuseppe"

#set document(title: title, author: author)


#show link: set text(rgb("#cc0052"))

#show ref: set text(green)

#set page(margin: (y: 0.5cm))
#set page(margin: (x: 1cm))

#set text(12pt)

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

//Codly
#show: codly-init.with()


#codly(
  languages: (
    rust: (name: "Rust", icon: "🦀", color: rgb("#CE412B")),
    py: (name: "Python", icon: "🐍", color: rgb("#4CAF50")),
    sh: (name: "Shell", icon: "💲", color: rgb("#89E051")),
  )
)



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


= Pyhton Cryptography

- Cryptography is a Python package that provides cryptographic recipes and primitives to developers
- It includes both high-level recipes and low-level interfaces to common cryptographic algorithms
- You can install cryptography with:
  ```sh pip install cryptography
  ``` 
  *N.B.* on kali linux it should be preinstalled.

== Fernet - The Recipes Layer
- It includes safe cryptographic recipes that require minimum choices
- Developers don’t make many decisions
- Implementation of symmetric authenticated cryptography
- It uses *AES* in *CBC* mode with 128-bit key for encryption and *PKCS7* padding
- It employs *HMAC* using *SHA256* for authentication
- Initialization vectors are generated using `os.urandom()`

== The Fernet Class
This is how the Fernet Class should be used
#figure(
  //caption: "Fernet usage"
)[
  #codly(header: [#align(center)[*Fernet usage*]])
  ```py
  >>> from cryptography.fernet import Fernet
  >>> key = Fernet.generate_key()
  >>> f = Fernet(key)
  >>> token = f.encrypt(b"my deep dark secret")
  >>> token
  b'...'
  >>> f.decrypt(token)
  b'my deep dark secret'
  ```
] <Fernet-usage>

The `key` parameter must be kept safe since the encrypted message contains the current time when it was generated, and so the time message will be visible to possible attackers.\
Indeed the current time can be extraced with `key.extract_timestamp(token)`.


=== Time-Based Security
With Fernet is possible to set a time expiration for the token created:

- ```py encrypt_at_time(token, current_time)``` - Encrypts data with a specific timestamp
- ```py decrypt_at_time(token, ttl, current_time)``` - Decrypts only if token hasn't exceeded its TTL (Time To Live)

Here's an example:

#figure(
  //caption: ""
)[
  #codly(header: [#align(center)[*Time-Based Security - Generating token*]])
  ```py
  from cryptography.fernet import Fernet
  from datetime import datetime, timedelta

  # Generate encryption key
  key = Fernet.generate_key()
  f = Fernet(key)

  # Security code sent at 9:00 AM
  send_time = datetime(2025, 5, 28, 9, 0, 0)  # 9:00 AM
  secret_message = b"Your verification code is: 847291. Use within 5 minutes."

  # Encrypt message with timestamp
  encrypted_token = f.encrypt_at_time(
      secret_message, 
      int(send_time.timestamp())
  )

  print(f"🔒 Message encrypted at: {send_time}")
  print(f"📱 Encrypted token: {encrypted_token[:50]}...")
  ```
] <Time-Based-Security-Generating-token>


#figure(
)[
  #codly(header: [#align(center)[*Time-Based Security - User decrypt the message*]])
  ```py
  # User tries to decrypt at 9:03 AM (3 minutes later)
  user_access_time = datetime(2025, 5, 28, 9, 3, 0)  # 9:03 AM
  ttl = 300  # 5 minutes = 300 seconds

  try:
      # Decrypt the message
      decrypted_message = f.decrypt_at_time(
          encrypted_token,
          ttl=ttl,
          current_time=int(user_access_time.timestamp())
      )
      
      print(f"✅ SUCCESS at {user_access_time.strftime('%I:%M %p')}")
      print(f"📄 Message: {decrypted_message.decode()}")
      print(f"⏰ Message age: 3 minutes (within 5-minute limit)")
      
  except Exception as e:
      print(f"❌ FAILED: {e}")
  ```
] <Time-Based-Security-User-decrypt-the-message>

== The MultiFernet Class
This is how MultiFernet should be used:
#figure(
  //caption: "MultiFernet usage"
)[
  #codly(header: [#align(center)[*MultiFernet usage*]])
  ```py
  >>> from cryptography.fernet import Fernet, MultiFernet
  >>> key1 = Fernet(Fernet.generate_key())
  >>> key2 = Fernet(Fernet.generate_key())
  >>> f = MultiFernet([key1, key2])
  >>> token = f.encrypt(b"Secret message!")
  >>> token
  b'...'
  >>> f.decrypt(token)
  b'Secret message!'

  ```
] <MultiFernet-usage>

The `MultiFernet` class extends `Fernet` allowing the management of multiple Fernet keys. This is paramount because, in any cryptographic system, keys should not be immortal. The principle of key rotation - periodically changing the keys used to encrypt data - is a cornerstone of good security practice. It serves to limit the potential damage if a key is compromised and to increase the difficulty of certain attacks.

This brings us to the `rotate` method. Imagine you have a corpus of data, all encrypted with a particular Fernet key. Perhaps this key is old, or worse, you suspect it may have been exposed – an employee departure, for instance, is a common trigger for such concerns. Simply starting to encrypt new data with a new key is insufficient; the old, potentially vulnerable data remains encrypted with the compromised key.

The `rotate` function elegantly addresses this. When you instantiate `MultiFernet`, you provide it with a list of Fernet objects (each initialized with a specific key). The key at the head of this list (the first one) is considered the primary key and is used for all new encryption operations.

#figure(
  //caption: "Rotate usage"
)[
  #codly(header: [#align(center)[*Rotate usage*]])
  ```py
  >>> from cryptography.fernet import Fernet, MultiFernet
  >>> key1 = Fernet(Fernet.generate_key())
  >>> key2 = Fernet(Fernet.generate_key())
  >>> f = MultiFernet([key1, key2])
  >>> token = f.encrypt(b"Secret message!")
  >>> token
  b'...'
  >>> f.decrypt(token)
  b'Secret message!'
  >>> key3 = Fernet(Fernet.generate_key())
  >>> f2 = MultiFernet([key3, key1, key2])
  >>> rotated = f2.rotate(token)
  >>> f2.decrypt(rotated)
  b'Secret message!'
  ```
]

When you call `rotate(token)` on a `MultiFernet` instance, the following occurs:

+ *Decryption:* `MultiFernet` will iterate through its list of keys and attempt to decrypt the `token`. It will use the key that successfully decrypts the token.
+ *Re-encryption:* Once decrypted, the plaintext data is then immediately re-encrypted using the primary key.
+ *Timestamp Preservation:* The original timestamp embedded within the Fernet token is preserved during this re-encryption process. This is important for maintaining the integrity of the token's age information.

The `rotate` method, therefore, allows you to systematically re-encrypt your existing data under a new, secure key without ever exposing the plaintext data during the transition.


== Using Password with Fernet
To use passwords securely with Fernet, the password needs to be transformed into a strong cryptographic key. Doing this require the use of a Key Derivation Function (KDF).

A KDF takes a password (and other parameters) and derives a cryptographically strong key from it. This process is also known as _"key stretching"_.

#figure()[
  #codly(header: [#align(center)[*Password management*]])
  ```py
  >>> import base64
  >>> import os
  >>> from cryptography.fernet import Fernet
  >>> from cryptography.hazmat.primitives import hashes
  >>> from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
  >>> password = b"password"
  >>> salt = os.urandom(16)
  >>> kdf = PBKDF2HMAC(
  ...    algorithm=hashes.SHA256(),
  ...    length=32,
  ...    salt=salt,
  ...    iterations=1_200_000,
  ... )
  >>> key = base64.urlsafe_b64encode(kdf.derive(password))
  >>> f = Fernet(key)
  >>> token = f.encrypt(b"Secret message!")
  >>> token
  b'...'
  >>> f.decrypt(token)
  b'Secret message!'
  ```
]

This "script" shows the use of `PBKDF2HMAC` (Password-Based Key Derivation Function 2 with HMAC). Other common KDFs include Argon2id and Scrypt.

- *PBKDF2HMAC:* Applies a pseudorandom function (like HMAC-SHA256) to the password and salt repeatedly.
- *Argon2id:* The winner of the Password Hashing Competition (2015), designed to be resistant to various attacks, including those using GPUs. It's often recommended for new applications.
- *Scrypt:* Designed to be memory-hard, making it expensive for attackers to perform large-scale custom hardware attacks.

The `salt` need to be *retrievable* because he must be used again with the password to re-derive the same key for decryption.

The `iterations` variable represents the number of times the KDF repeatedly applies its internal hashing function. A higher iteration count implies that more computational resources will be required by an ideal hacker performing a brute force attack.


== Symmetric Encryption

`Cipher` objects combine an algorithm such as AES with a mode like CBC or CTR. A simple example of encrypting and then decrypting content with AES is:

#figure()[
  #codly(header: [#align(center)[*Example of Encrypting and Decrypting*]])
  ```py
  >>> import os
  >>> from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
  >>> key = os.urandom(32)
  >>> iv = os.urandom(16)
  >>> cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
  >>> encryptor = cipher.encryptor()
  >>> ct = encryptor.update(b"a secret message") + encryptor.finalize()
  >>> decryptor = cipher.decryptor()
  >>> decryptor.update(ct) + decryptor.finalize()
  b'a secret message'
  ```
]
