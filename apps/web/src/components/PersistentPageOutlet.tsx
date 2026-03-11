import { useEffect, useMemo, useState } from "react";
import { Navigate, useLocation } from "react-router-dom";

import InferencePage from "../pages/InferencePage";
import PostProcessingPage from "../pages/PostProcessingPage";
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
        title="Training (WIP)"
        description="To be implemented"
      />
    ),
  },
  { path: "/inference", render: () => <InferencePage /> },
  { path: "/post-processing", render: () => <PostProcessingPage /> },
];

const knownPaths = new Set(pageDefinitions.map((page) => page.path));
const legacyRedirects = new Map<string, string>([
  ["/segmentation", "/post-processing"],
  ["/tracking", "/post-processing"],
]);
const defaultPath = "/inference";

function normalizePathname(pathname: string): string {
  if (pathname === "/") return pathname;
  const normalized = pathname.replace(/\/+$/, "");
  return normalized.length > 0 ? normalized : "/";
}

export default function PersistentPageOutlet() {
  const location = useLocation();
  const activePath = normalizePathname(location.pathname);
  const redirectedPath = legacyRedirects.get(activePath);
  const resolvedActivePath = redirectedPath ?? activePath;
  const isKnownPath = knownPaths.has(resolvedActivePath);
  const [visitedPaths, setVisitedPaths] = useState<Set<string>>(() =>
    new Set<string>(isKnownPath ? [resolvedActivePath] : [defaultPath]),
  );

  useEffect(() => {
    if (!isKnownPath) return;
    setVisitedPaths((prev) => {
      if (prev.has(resolvedActivePath)) return prev;
      const next = new Set(prev);
      next.add(resolvedActivePath);
      return next;
    });
  }, [isKnownPath, resolvedActivePath]);

  const pagesToRender = useMemo(
    () => pageDefinitions.filter((definition) => visitedPaths.has(definition.path)),
    [visitedPaths],
  );

  if (redirectedPath) {
    return <Navigate to={redirectedPath} replace />;
  }

  if (!isKnownPath) {
    return <Navigate to={defaultPath} replace />;
  }

  return (
    <>
      {pagesToRender.map((page) => {
        const active = page.path === resolvedActivePath;
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
