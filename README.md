PJM Hand Recognition
Projekt rozpoznawania liter PJM (Polski Język Migowy) w czasie rzeczywistym przy użyciu MediaPipe i sieci neuronowych.

Jak możesz pomóc?
Potrzebuje nagrań gestów dla wszystkich liter PJM. Im więcej osób nagra dane, tym dokładniejszy będzie model. 
Optymalna odległość: 50–80 cm od kamery (mniej więcej długość wyciągniętego ramienia). Pozycja: kamera na poziomie klatki piersiowej. Idealnie: 40-60 próbek na literę nagrywane najlepiej na przestrzeni kilku dni, by ułożenie ręki było różne na każdym nagraniu, ale oczywiście dostosuj do swoich możliwości. Za każdą pomoc dziękuję <3.

Wymagania
Python 3.10+
Kamera internetowa
Instalacja: git clone git@github.com:sarnavka/pjm-hand-recognition.git
cd pjm-hand-recognition
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Filmiki i przydatne strony pokazujace jak poprawnie migać:
https://www.youtube.com/watch?v=xpisaLwDmn4,
https://www.youtube.com/watch?v=1UofccUa3U0,
https://www.youtube.com/watch?v=RrNBxkB3pyc,
https://www.pzg.szczecin.pl/multimedialny-slownik-jezyka-migowego/?1&cale#home,
https://spreadthesign.com/pl.pl/alphabet/29/,
https://www.youtube.com/watch?v=tybvooQblkc&list=PL6bKyVNfhEbacK3Z2hXVoyYUeyLWRDZXz&index=9

python collect_pjm.py
