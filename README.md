# PJM Hand Recognition

Projekt rozpoznawania liter PJM (Polski Język Migowy) w czasie rzeczywistym przy użyciu MediaPipe i sieci neuronowych.

## Jak możesz pomóc?

Potrzebuję nagrań gestów dla wszystkich liter PJM. Im więcej osób nagra dane, tym dokładniejszy będzie model.

- **Optymalna odległość:** 50–80 cm od kamery (mniej więcej długość wyciągniętego ramienia)
- **Pozycja:** kamera na poziomie klatki piersiowej, jednolite tło
- **Ile nagrywać:** idealnie 40 próbek na literę od Ciebie (w aplikacji zobaczysz pasek postępu z celem 120 — to łączny cel projektu, sumowany z nagrań kilku osób, nie musisz dobić do niego sam)
- Najlepiej nagrywać na przestrzeni kilku dni, np. przez 4 dni po 10 próbek na literę
- Zależy mi, żeby ułożenie ręki było trochę inne na każdym nagraniu (kąt, odległość, oświetlenie)
- Oczywiście dostosuj do swoich możliwości — każda pomoc się liczy

Za każdą pomoc dziękuję <3

## Wymagania

- Python 3.10–3.11 (nowsze wersje mogą mieć problem z instalacją MediaPipe)
- Kamera internetowa

## Instalacja

git clone https://github.com/sarnavka/pjm-hand-recognition.git
cd pjm-hand-recognition
python -m venv venv


Aktywacja środowiska:
- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

Instalacja zależności (tylko to, co potrzebne do nagrywania — bez ciężkiego TensorFlow):

## Jak poprawnie migać

Filmiki i przydatne strony:
- https://www.youtube.com/watch?v=xpisaLwDmn4
- https://www.youtube.com/watch?v=1UofccUa3U0
- https://www.youtube.com/watch?v=RrNBxkB3pyc
- https://www.pzg.szczecin.pl/multimedialny-slownik-jezyka-migowego/?1&cale#home
- https://spreadthesign.com/pl.pl/alphabet/29/
- https://www.youtube.com/watch?v=tybvooQblkc&list=PL6bKyVNfhEbacK3Z2hXVoyYUeyLWRDZXz&index=9

## Nagrywanie

python collect_pjm.py

W menu wybierz **S** dla liter statycznych albo **D** dla liter dynamicznych (z ruchem) — reszta sterowania (SPACJA = zapisz, R = usuń ostatnią, N = następna litera, B = menu, Q = wyjście) jest opisana na ekranie.

## Wysyłanie danych

Po skończonych nagraniach spakuj folder `data` (podfoldery `pjm_dataset` i `pjm_sequences`) do ZIP-a i wyślij mi go np. przez Google Drive — sam plik jest za duży/nieobsługiwany przez e-mail i nie da się go po prostu wypchnąć przez git.
