# TomExplorer

TomExplorer ist eine browserbasierte GUI fuer Absorptionsspektren. Die App ersetzt den bisherigen Notebook-Workflow durch zwei direkte Arbeitsmodi:

- Manuelle Spektren mit Dropdown-Auswahl, Konzentrationen, Temperatur, Druck, Bereichseingabe und Hover-Detailpanel fuer sigma und alpha.
- Automatisierte Bandensuche mit Zielgasen, Stoergasen, maximaler Laseranzahl und klickbarer Trefferliste fuer vorgeschlagene Durchstimmbereiche.

## Start

```bash
pip install -r requirements.txt
python tomexplorer_app.py
```

Danach oeffnet die App lokal im Browser unter der von Dash ausgegebenen Adresse.

## Hinweise

- VS Code soll mit dem globalen Python laufen. Eine versehentlich mitkopierte `.venv` ist fuer dieses Projekt nicht noetig und kann ignoriert werden.
- Die manuelle Spektrenberechnung und die Bandensuche laufen temperatur- und druckabhaengig direkt ueber die lokale HAPI/HITRAN-Datenbank in `hitran_cache/`.
- Die Schaltflaeche fuer den HITRAN-Refresh laedt die gewaehlten Gase neu, baut die Offline-Datei `abscross_dict.pkl` fuer den Bereich neu auf und raeumt alte ungenutzte Hilfsdateien weg.
- `abscross_dict.pkl` bleibt als Offline-Export erhalten, ist aber nicht mehr die Quelle fuer die aktuellen Live-Plots.
- Der erste Zugriff auf ein Gas oder einen neuen Bereich kann spuerbar dauern, weil TomExplorer die benoetigten HITRAN-Daten erst lokal laden muss.
- Die Startmeldung zu HAPI2 kommt direkt aus der mitgelieferten HAPI-Bibliothek. Sie ist nur ein Hinweis des HITRAN-Teams auf eine erweiterte Zusatzbibliothek und kein Fehler.
- Sehr grosse Spektralbereiche koennen rechenintensiv sein. In der GUI kann die Schrittweite groesser gewaehlt werden, um die Suche zu beschleunigen.
- Fuer die Plot-Performance werden sehr dichte Spektren automatisch auf eine browserfreundliche Punktzahl heruntergebrochen. Die physikalische Berechnung erfolgt davor auf dem gewaehlten Gitter.
