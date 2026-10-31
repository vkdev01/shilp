// Theme: follow the OS until the user picks one, then remember the choice.
// Runs in <head> so the right palette is in place before first paint.
(function () {
  "use strict";

  var KEY = "shilp-theme";
  var root = document.documentElement;
  var media = window.matchMedia("(prefers-color-scheme: dark)");

  function stored() {
    try {
      var value = localStorage.getItem(KEY);
      return value === "light" || value === "dark" ? value : null;
    } catch (_) {
      return null;
    }
  }

  function apply(theme) {
    root.dataset.theme = theme;
  }

  function current() {
    return stored() || (media.matches ? "dark" : "light");
  }

  apply(current());

  media.addEventListener("change", function () {
    if (!stored()) apply(current());
  });

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-theme-toggle]").forEach(function (button) {
      button.addEventListener("click", function () {
        var next = root.dataset.theme === "dark" ? "light" : "dark";
        try {
          localStorage.setItem(KEY, next);
        } catch (_) {
          /* private mode: the toggle still works for this page */
        }
        apply(next);
      });
    });
  });
})();
