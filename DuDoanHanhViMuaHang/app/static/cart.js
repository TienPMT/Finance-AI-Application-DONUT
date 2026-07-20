const storageKey = "kpdl-cart-item-ids";

function readCart() {
  try {
    const values = JSON.parse(localStorage.getItem(storageKey) || "[]");
    return [...new Set(values.filter((value) => Number.isInteger(value) && value > 0))];
  } catch {
    return [];
  }
}

function writeCart(itemIds) {
  localStorage.setItem(storageKey, JSON.stringify(itemIds));
  updateCartLinks(itemIds);
}

function updateCartLinks(itemIds = readCart()) {
  document.querySelectorAll("[data-cart-count]").forEach((node) => {
    node.textContent = String(itemIds.length);
  });
  document.querySelectorAll("[data-cart-link]").forEach((node) => {
    node.href = itemIds.length ? `/cart?items=${itemIds.join(",")}` : "/cart";
  });
}

function announce(message) {
  const status = document.querySelector("[data-cart-status]");
  if (!status) return;
  status.textContent = message;
  status.classList.add("visible");
  window.setTimeout(() => status.classList.remove("visible"), 2200);
}

function addItem(button) {
  const itemId = Number(button.dataset.addToCart);
  const itemIds = readCart();
  if (!itemIds.includes(itemId)) itemIds.push(itemId);
  writeCart(itemIds);
  announce(`Đã thêm ${button.dataset.productName || "sản phẩm"} vào giỏ.`);
  if (document.querySelector("[data-cart-page]")) {
    window.location.assign(`/cart?items=${itemIds.join(",")}`);
  }
}

function removeItem(itemId) {
  const itemIds = readCart().filter((value) => value !== itemId);
  writeCart(itemIds);
  window.location.assign(itemIds.length ? `/cart?items=${itemIds.join(",")}` : "/cart");
}

document.addEventListener("click", (event) => {
  const addButton = event.target.closest("[data-add-to-cart]");
  if (addButton) addItem(addButton);

  const removeButton = event.target.closest("[data-remove-from-cart]");
  if (removeButton) removeItem(Number(removeButton.dataset.removeFromCart));

  if (event.target.closest("[data-clear-cart]")) {
    writeCart([]);
    window.location.assign("/cart");
  }
});

document.addEventListener("DOMContentLoaded", () => {
  const queryItems = new URLSearchParams(window.location.search).get("items");
  if (document.querySelector("[data-cart-page]") && queryItems !== null) {
    const itemIds = queryItems
      .split(",")
      .map(Number)
      .filter((value) => Number.isInteger(value) && value > 0);
    writeCart([...new Set(itemIds)]);
  } else {
    updateCartLinks();
  }
});
