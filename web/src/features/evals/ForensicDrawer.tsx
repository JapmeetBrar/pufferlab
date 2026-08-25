import { type KeyboardEvent, type ReactNode, type RefObject, useEffect, useRef } from "react";

function focusableElements(container: HTMLElement): HTMLElement[] {
  return Array.from(
    container.querySelectorAll<HTMLElement>(
      'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    ),
  ).filter((element) => !element.hasAttribute("hidden"));
}

export function ForensicDrawer({
  documentLabel,
  onClose,
  returnFocusRef,
  children,
}: {
  documentLabel: string;
  onClose: () => void;
  returnFocusRef: RefObject<HTMLElement | null>;
  children: ReactNode;
}) {
  const drawerRef = useRef<HTMLElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    const returnTarget = returnFocusRef.current;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();
    return () => {
      document.body.style.overflow = previousOverflow;
      const target = returnTarget ?? document.getElementById("query-forensics-heading");
      target?.focus();
    };
  }, [returnFocusRef]);

  function handleKeyDown(event: KeyboardEvent<HTMLElement>) {
    if (event.key === "Escape") {
      event.preventDefault();
      onClose();
      return;
    }
    if (event.key !== "Tab" || drawerRef.current === null) return;
    const focusable = focusableElements(drawerRef.current);
    const first = focusable[0];
    const last = focusable.at(-1);
    if (first === undefined || last === undefined) return;
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  return (
    <div className="drawer-backdrop">
      <aside
        ref={drawerRef}
        className="forensic-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="forensic-drawer-heading"
        onKeyDown={handleKeyDown}
      >
        <header className="drawer-header">
          <div>
            <p className="eyebrow">Exact document target</p>
            <h2 id="forensic-drawer-heading">Document evidence</h2>
            <p className="drawer-document-title">{documentLabel}</p>
          </div>
          <button ref={closeRef} type="button" onClick={onClose} aria-label="Close document evidence">
            Close
          </button>
        </header>
        <div className="drawer-body">{children}</div>
      </aside>
    </div>
  );
}
