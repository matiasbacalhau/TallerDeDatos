-- Las imágenes que la notebook muestra con un ancho fijo en píxeles (p. ej. Image(..., width=820))
-- se salen de la página A4: se ajustan al ancho del texto.
function Image(el)
  local w = el.attributes["width"]
  if w and not w:match("%%$") then
    el.attributes["width"] = "100%"
  end
  return el
end
