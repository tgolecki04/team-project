document.getElementById("hero").addEventListener("click", function() {
  const target = document.getElementById("start");   // element docelowy
  const offset = 50;                                  // dodatkowe 50px

  const topPosition = target.getBoundingClientRect().top + window.scrollY - offset;

  window.scrollTo({
    top: topPosition,
    behavior: "smooth"
  });
});