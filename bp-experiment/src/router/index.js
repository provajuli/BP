import { createRouter, createWebHashHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import ExperimentView from '../views/ExperimentView.vue'

export default createRouter({
    history: createWebHashHistory(),
    routes: [
        { path: '/', component: HomeView },
        { path: '/experiment', component: ExperimentView },
    ],
})
