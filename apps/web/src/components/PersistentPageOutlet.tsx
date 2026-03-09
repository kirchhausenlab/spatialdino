import { useEffect, useMemo, useState } from "react";
import { Navigate, useLocation } from "react-router-dom";

import InferencePage from "../pages/InferencePage";
import PlaceholderPage from "../pages/PlaceholderPage";

type PageDefinition = {
  path: string;
  render: () => JSX.Element;
};

const pageDefinitions: PageDefinition[] = [
  {
    path: "/training",
    render: () => (
      <PlaceholderPage
        title="Training"
        description="Training controls will be added here once the spatialDINO workflows are wired into the backend."
      />
    ),
  },
  { path: "/inference", render: () => <InferencePage /> },
  {
    path: "/segmentation",
    render: () => (
      <PlaceholderPage
        title="Segmentation"
        description="Segmentation tools are reserved here and will be connected in a later pass."
      />
    ),
  },
  {
    path: "/tracking",
    render: () => (
      <PlaceholderPage
        title="Tracking"
        description="Tracking workflows are reserved here and will be connected in a later pass."
      />
    ),
  },
];

const knownPaths = new Set(pageDefinitions.map((page) => page.path));
const defaultPath = "/inference";

function normalizePathname(pathname: string): string {
  if (pathname === "/") return pathname;
  const normalized = pathname.replace(/\/+$/, "");
  return normalized.length > 0 ? normalized : "/";
}

export default function PersistentPageOutlet() {
  const location = useLocation();
  const activePath = normalizePathname(location.pathname);
  const isKnownPath = knownPaths.has(activePath);
  const [visitedPaths, setVisitedPaths] = useState<Set<string>>(() =>
    new Set<string>(isKnownPath ? [activePath] : [defaultPath]),
  );

  useEffect(() => {
    if (!isKnownPath) return;
    setVisitedPaths((prev) => {
      if (prev.has(activePath)) return prev;
      const next = new Set(prev);
      next.add(activePath);
      return next;
    });
  }, [activePath, isKnownPath]);

  const pagesToRender = useMemo(
    () => pageDefinitions.filter((definition) => visitedPaths.has(definition.path)),
    [visitedPaths],
  );

  if (!isKnownPath) {
    return <Navigate to={defaultPath} replace />;
  }

  return (
    <>
      {pagesToRender.map((page) => {
        const active = page.path === activePath;
        return (
          <div
            key={page.path}
            className={active ? "persistentPage isActive" : "persistentPage"}
            aria-hidden={!active}
          >
            {page.render()}
          </div>
        );
      })}
    </>
  );
}
