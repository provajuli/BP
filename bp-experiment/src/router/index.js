import { createRouter, createWebHashHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import ExperimentView from '../views/ExperimentView.vue'
import CalibrationView from '../views/CalibrationView.vue'

export default createRouter({
    history: createWebHashHistory(),
    routes: [
        { path: '/', component: HomeView },
        { path: '/experiment', component: ExperimentView },
        { path: '/calibration', component: CalibrationView},
    ],
})
