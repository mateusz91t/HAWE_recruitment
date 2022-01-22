# Zadanie rekrutacyjne z HAWE Telekom

Do zadań użyłem głównie pandas. Wykresy do analizy danych w matplotlib i seaborn.

## Zadanie 1
Plik tekstowy `zadanie 1 wnioski.txt` zawiera rozwiązanie zadania nr 1, analiza i wyciągnięcie wniosków na podstawie pliku `functions_and_data_cleaning_dir\checking_data.py`. Plik ten uruchamiany był w IPython linia po linii więc nie nadaje się do uruchomienia jak skrypt. 2 ostatnie wykresy ładują się bardzo długo, ale były bardzo przydatne do analizy.

## Zadanie 2
Plik pythonowy `functions_and_data_cleaning_dir\moving_average_using.py` zawiera użycie funkcji średniej kroczącej. Funkcja jest w pliku `functions_and_data_cleaning_dir\additional_functions.py`. W zadaniu nie było informacji, czy mam od podstaw ją stworzyć, więc nie programowałem jej w całości sam - skorzystałem z istniejącej w pandas metody rolling. Jeśli to nie jest akceptowalne rozwiązanie proszę o mail/tel w celu poprawienia. Mam wątpliwość, czy na pewno każdy wiersz powinien zawierać średnią, czy tak jak robi rolling - dla 5cio wierszowego okna dopiero 5ty wiersz ma średnią ostatnich 5ciu wierszów, a poprzednie wiersze to NaN. Do funkcji napisałem też 2 przypaki testowe i funkcję testującą w pytest: `functions_and_data_cleaning_dir\tests\test_additional_functions.py`.

## Zadanie 3 - wykonane po czasie dla siebie jako ciekawostka
sklearn.linear_model.LinearRegression + one hot encoding mapping in v_1 parameter
How to do a forecast for time series? Standard extrapolation? Libs?

https://www.cienciadedatos.net/documentos/py27-time-series-forecasting-python-scikitlearn.html
If I want to forecast a label target in the future base on knows attributes without submit these attributes, I need to extrapolate X labels and base on it predict target label.
libs: sklearn, skforecast
