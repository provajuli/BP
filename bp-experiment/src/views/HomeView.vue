<template>
  <section>
    <h1>O bakalářské práci</h1>

    <p>
      Tato webová aplikace slouží jako demonstrátor a experimentální prostředí k mé bakalářské práci.
      V experimentu uživatel porovnává vizuální stimuly (glyphy) a upravuje jejich parametr tak,
      aby odpovídal referenčním hodnotám.
    </p>

    <h2>Jak experiment probíhá</h2>
    <ol>
      <li>Na obrazovce se zobrazí tři glyphy: A, B a C.</li>
      <li>A a C mají pevnou velikost, B upravuješ kolečkem myši (krok 1).</li>
      <li>Tvým úkolem je nastavit B tak, aby subjektivně odpovídal mezi A a C.</li>
      <li>Po kliknutí na „Next“ se uloží anonymní výsledky do CSV.</li>
    </ol>

    <h2>Anonymita a data</h2>
    <p>
      Nejsou ukládány identifikátory jako jméno, e-mail ani přihlašovací údaje.
      Výsledky se ukládají anonymně do souboru CSV pro každou session.
    </p>

    <div class="cta">
      <RouterLink class="btn" to="/experiment">Spustit pokus</RouterLink>
    </div>
  </section>
</template>

<style scoped>
h1 { margin: 0 0 12px; }
h2 { margin-top: 18px; }
.cta { margin-top: 18px; }
.btn { display: inline-block; padding: 10px 14px; border-radius: 10px; background: #111; color: #fff; text-decoration: none; }
</style>
