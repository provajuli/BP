import { createI18n } from "vue-i18n";
import en from "./lang/en/en.json";
import cs from "./lang/cs/cs.json";

const i18n = createI18n({
    legacy: false,
    locale: localStorage.getItem("lang") || "en",
    fallbackLocale: "en",
    messages: {
        en: en,
        cs: cs,
    },
});

export default i18n;