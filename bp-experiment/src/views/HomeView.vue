<template>
  <section>
    <h1>O bakalářské práci</h1>

    <p>
      Tato webová aplikace slouží jako experimentální prostředí pro sběr dat k mé bakalářské práci na téma 
      <b>Weberův-Fechnerův zákon ve spojitosti s Malleable Glyphs</b>. 
    </p>

    <h2>Jak experiment probíhá</h2>
    <ol>
      <li>Na obrazovce se zobrazí tři glyphy: A, B a C (A a C mohou mít libovolné vzájemné pořadí velikostí).</li>
      <li>A a C mají pevnou velikost, B upravujete kolečkem myši (příp. touchpadem).</li>
      <li>Vaším úkolem je nastavit B tak, aby subjektivně odpovídalo <b>středu velikostí mezi A a C</b>.</li>
      <li>Po kliknutí na „Next“ se uloží anonymní výsledky do souboru CSV.</li>
    </ol>

    <div class="calib">
      <p>
        Pro správnou fyzickou velikost glyphů (1x1 inch) doporučuji provést kalibraci monitoru.
      </p>

      <RouterLink class="btn secondary" to="/calibration">Kalibrace monitoru</RouterLink>
    </div>

    <h2>Časová náročnost</h2>
    <p>
      Délka experimentu není pevně stanovena a závisí na každém, kolik času mu věnuje.
      Experiment lze kdykoliv ukončit a jakýkoliv počet dokončených úloh je pro mě přínosný.
    </p>

    <h2>Anonymita a data</h2>
    <p>
      Výsledky se ukládají anonymně do souboru CSV pro každou session.
    </p>

    <h2>Další informace</h2>
    <p>
      Pokud vás téma <i>Malleable Glyphs</i> zaujalo a chcete se o něm dozvědět více:
    </p>
    <ul>
      <li>
        <a href="https://tmgc.fit.vutbr.cz/#what-is-a-malleable-glyph" target="_blank" rel="noopener">
          The Malleable Glyph Challenge 
        </a>
      </li>
      <li>
        <a href="https://github.com/adamherout/mglyph" target="_blank" rel="noopener">
          GitHub repozitář knihovny mglyph pro Python
        </a>
      </li>
    </ul>

    <div class="cta">
      <RouterLink class="btn" to="/experiment">Spustit pokus</RouterLink>
    </div>
  </section>
</template>

<style scoped>
section {
  max-width: 720px;
  line-height: 1.55;
}

/* Základní text */
p, li {
  font-size: 15px;
  color: #222;
}

/* Nadpisy */
h1 {
  margin: 0 0 10px;
  font-size: 26px;
  font-weight: 700;
}

h2 {
  margin-top: 20px;
  margin-bottom: 6px;
  font-size: 18px;
  font-weight: 600;
}

/* Seznam */
ol {
  padding-left: 20px;
}
li {
  margin-bottom: 6px;
}

/* Kalibrace box */
.calib {
  margin: 16px 0;
  padding: 12px 14px;
  background: #f6f6f6;
  border-radius: 10px;
  font-size: 14px;
}

/* CTA */
.cta {
  margin-top: 22px;
}

/* Tlačítka */
.btn {
  display: inline-block;
  padding: 8px 12px;
  border-radius: 8px;
  font-size: 14px;
  background: #111;
  color: #fff;
  text-decoration: none;
}

.btn.secondary {
  background: #fff;
  color: #111;
  border: 1px solid #111;
  margin-left: 8px;
}

</style>
