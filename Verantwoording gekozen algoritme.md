In onze applicatie hebben wij gekozen voor een CNN. 
De reden hiervoor is dat wanneer de camera verplaatst wordt (bij het kuisen bijvoorbeeld of omstoten in het passeren),
je niet telkens het model moet gaan trainen. 
Het CNN scoort iets minder goed in percentages (slecht een paar %), wat nog steeds een heel mooi resultaat geeft van 93% ten opzichte van 95% voor een RandomForrest.
Het grote voordeel tov een RandomForrest is zoals eerder vermeld, de camera kan onder een lichte andere hoek geplaatst worden, het zal nog steeds werken dankzij de maxPooling2D.
Verder hebben we deze architectuur gekozen omdat deze goed scoorde bij andere toepassingen.
Er is ook gekeken om een LeNet-5 architectuur toe te passen, maar deze scoorde bijzonder slecht op project Musti.
s