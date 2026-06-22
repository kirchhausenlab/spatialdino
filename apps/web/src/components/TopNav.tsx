import { NavLink } from "react-router-dom";

import ThemeToggle from "./ThemeToggle";

const navItems: Array<{ to: string; label: string }> = [
  { to: "/data", label: "Data" },
  { to: "/training", label: "Training" },
  { to: "/inference", label: "Inference" },
  { to: "/post-processing", label: "Post-processing" },
  { to: "/segmentation", label: "Segmentation" },
  { to: "/tracking", label: "Tracking" }
];

export default function TopNav() {
  return (
    <header className="topNav">
      <nav className="topNavInner" aria-label="Primary">
        <div className="topNavLeft">
          <NavLink to="/data" className="topNavBrand" aria-label="SpatialDINO home">
            SpatialDINO
          </NavLink>
          <div className="topNavDivider" aria-hidden="true" />
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => (isActive ? "topNavLink isActive" : "topNavLink")}
            >
              {item.label}
            </NavLink>
          ))}
        </div>
        <div className="topNavRight">
          <ThemeToggle />
        </div>
      </nav>
    </header>
  );
}
