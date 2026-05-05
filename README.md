PJM Hand Recognition
Projekt rozpoznawania liter daktylografii PJM (Polski Język Migowy) w czasie rzeczywistym przy użyciu MediaPipe i sieci neuronowych.

Jak możesz pomóc?
Potrzebujemy nagrań gestów dla wszystkich liter PJM. Im więcej osób nagra dane, tym dokładniejszy będzie model.

Wymagania
Python 3.10+
Kamera internetowa
Instalacja

git clone git@github.com:sarnavka/pjm-hand-recognition.git
cd pjm-hand-recognition
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
Zbieranie danych

Filmiki i przydatne strony pokazujace jak poprawnie migać:
https://www.youtube.com/watch?v=xpisaLwDmn4
https://www.youtube.com/watch?v=1UofccUa3U0
https://www.youtube.com/watch?v=RrNBxkB3pyc
https://www.pzg.szczecin.pl/multimedialny-slownik-jezyka-migowego/?1&cale#home
https://spreadthesign.com/pl.pl/alphabet/29/
https://www.youtube.com/watch?v=tybvooQblkc&list=PL6bKyVNfhEbacK3Z2hXVoyYUeyLWRDZXz&index=9

python collect_pjm.py
S — nagraj litery statyczne (bez ruchu)
D — nagraj litery dynamiczne (z ruchem)
I — sprawdź ile masz już nagrań
SPACJA — zapisz zdjęcie / zacznij nagrywanie
R — usuń ostatnie nagranie
Q — wyjdź

