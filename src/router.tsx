import { Outlet, createRootRoute, createRoute, createRouter } from '@tanstack/react-router';
import MainLayout from './layouts/MainLayout';
import Home from './pages/Home';
import ProjectDetail from './pages/ProjectDetail';
import CV from './pages/CV';

const rootRoute = createRootRoute({
  component: () => (
    <MainLayout>
      <Outlet />
    </MainLayout>
  ),
});

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: Home,
});

const projectRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/project/$id',
  component: ProjectDetail,
});

const cvRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/cv',
  component: CV,
});

const routeTree = rootRoute.addChildren([indexRoute, projectRoute, cvRoute]);

export const router = createRouter({
  routeTree,
  basepath: import.meta.env.BASE_URL,
});

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router;
  }
}
