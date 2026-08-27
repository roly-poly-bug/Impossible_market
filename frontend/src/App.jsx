import { useEffect, useState } from "react";


const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

function formatPrice(price) {
  const [integerPart, decimalPart = ""] = String(price).split(".");
  const formattedInteger = BigInt(integerPart).toLocaleString("ja-JP");
  const meaningfulDecimals = decimalPart.replace(/0+$/, "");
  return `¥${formattedInteger}${meaningfulDecimals ? `.${meaningfulDecimals}` : ""}`;
}

function formatRarity(rarity) {
  return `${(rarity * 100).toLocaleString("en", { maximumFractionDigits: 1 })}% rarity`;
}

function useApi(path) {
  const [result, setResult] = useState({ state: "loading", data: null, message: "" });

  useEffect(() => {
    const controller = new AbortController();
    setResult({ state: "loading", data: null, message: "" });

    async function load() {
      try {
        const response = await fetch(`${apiBaseUrl}${path}`, { signal: controller.signal });
        if (response.status === 404) {
          setResult({ state: "not-found", data: null, message: "This impossible item does not exist." });
          return;
        }
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        setResult({ state: "success", data: await response.json(), message: "" });
      } catch (error) {
        if (error.name !== "AbortError") {
          setResult({
            state: "error",
            data: null,
            message: "The catalog API is unreachable. Make sure the FastAPI server is running.",
          });
        }
      }
    }

    load();
    return () => controller.abort();
  }, [path]);

  return result;
}

function Header() {
  return (
    <header className="site-header">
      <a className="brand" href="/">Impossible Market</a>
      <span className="brand-note">Fictional goods. Real engineering.</span>
    </header>
  );
}

function Message({ children }) {
  return <div className="message" role="status">{children}</div>;
}

function ProductCard({ product }) {
  return (
    <article className="product-card">
      <div className="product-card__meta">
        <span>{product.category}</span>
        <span>{formatRarity(product.rarity)}</span>
      </div>
      <h2>{product.name}</h2>
      <p className="product-card__description">{product.description}</p>
      <div className="product-card__footer">
        <strong>{formatPrice(product.price)}</strong>
        <a href={`/products/${product.id}`} aria-label={`View ${product.name}`}>View artifact →</a>
      </div>
    </article>
  );
}

function CatalogPage() {
  const products = useApi("/api/products");

  return (
    <>
      <section className="catalog-hero">
        <p className="eyebrow">The impossible collection</p>
        <h1>Everything has a price.<br />Even the unthinkable.</h1>
        <p className="tagline">Browse singular artifacts, extinct wonders, and things no market should contain.</p>
      </section>

      {products.state === "loading" && <Message>Opening the vault…</Message>}
      {products.state === "error" && <Message>{products.message}</Message>}
      {products.state === "success" && products.data.length === 0 && (
        <Message>The catalog is empty. Run the product seed command to stock the shelves.</Message>
      )}
      {products.state === "success" && products.data.length > 0 && (
        <section className="product-grid" aria-label="Product catalog">
          {products.data.map((product) => <ProductCard key={product.id} product={product} />)}
        </section>
      )}
    </>
  );
}

function ProductDetailPage({ productId }) {
  const product = useApi(`/api/products/${productId}`);

  if (product.state === "loading") return <Message>Retrieving artifact…</Message>;
  if (product.state === "error" || product.state === "not-found") {
    return (
      <section className="detail-state">
        <Message>{product.message}</Message>
        <a className="back-link" href="/">← Return to the catalog</a>
      </section>
    );
  }

  return (
    <article className="product-detail">
      <a className="back-link" href="/">← All impossible goods</a>
      {product.data.image_url && (
        <img className="product-detail__image" src={product.data.image_url} alt={product.data.name} />
      )}
      <div className="product-detail__meta">
        <span>{product.data.category}</span>
        <span>{formatRarity(product.data.rarity)}</span>
      </div>
      <h1>{product.data.name}</h1>
      <p className="product-detail__description">{product.data.description}</p>
      <p className="product-detail__price">{formatPrice(product.data.price)}</p>
      <p className="availability">One known example · Acquisition unavailable</p>
    </article>
  );
}

export default function App() {
  const detailMatch = window.location.pathname.match(/^\/products\/(\d+)\/?$/);

  return (
    <div className="page-shell">
      <Header />
      <main>
        {detailMatch ? <ProductDetailPage productId={detailMatch[1]} /> : <CatalogPage />}
      </main>
    </div>
  );
}
