import { useEffect, useState } from "react";


const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const catalogPageSize = 24;

function formatPrice(price) {
  const [integerPart, decimalPart = ""] = String(price).split(".");
  const formattedInteger = BigInt(integerPart).toLocaleString("ja-JP");
  const meaningfulDecimals = decimalPart.replace(/0+$/, "");
  return `¥${formattedInteger}${meaningfulDecimals ? `.${meaningfulDecimals}` : ""}`;
}

function formatRarity(rarity) {
  return `${(rarity * 100).toLocaleString("en", { maximumFractionDigits: 1 })}% rarity`;
}

function formatStatus(status) {
  return status.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());
}

const publicAttributeLabels = {
  danger: "Danger",
  luxury: "Luxury",
  novelty: "Novelty",
  historical_value: "Historical value",
  technology_level: "Technology level",
  natural_significance: "Natural significance",
};

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
  const [visibleCount, setVisibleCount] = useState(catalogPageSize);

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
        <>
          <section className="product-grid" aria-label="Product catalog">
            {products.data
              .slice(0, visibleCount)
              .map((product) => <ProductCard key={product.id} product={product} />)}
          </section>
          <div className="catalog-progress">
            <span>Showing {Math.min(visibleCount, products.data.length)} of {products.data.length}</span>
            {visibleCount < products.data.length && (
              <button type="button" onClick={() => setVisibleCount((count) => count + catalogPageSize)}>
                Show more artifacts
              </button>
            )}
          </div>
        </>
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
        <span>{product.data.category.name}</span>
        <span>{formatRarity(product.data.rarity)}</span>
      </div>
      <h1>{product.data.name}</h1>
      <p className="product-detail__description">{product.data.description}</p>
      <p className="product-detail__price">{formatPrice(product.data.price)}</p>
      <p className={`availability availability--${product.data.status}`}>
        Status · {formatStatus(product.data.status)}
      </p>
      <p className="reality-type">Reality · {formatStatus(product.data.reality_type)}</p>

      <section className="metadata-section" aria-labelledby="tags-heading">
        <h2 id="tags-heading">Known qualities</h2>
        <div className="tag-list">
          {product.data.tags.map((tag) => <span className="tag" key={tag.id}>{tag.name}</span>)}
        </div>
      </section>

      <section className="metadata-section" aria-labelledby="attributes-heading">
        <h2 id="attributes-heading">Artifact profile</h2>
        <div className="attribute-list">
          {Object.entries(product.data.attributes)
            .filter(([name]) => publicAttributeLabels[name])
            .map(([name, value]) => (
              <div className="attribute" key={name}>
                <div className="attribute__label">
                  <span>{publicAttributeLabels[name]}</span>
                  <span>{Math.round(value * 100)}%</span>
                </div>
                <div className="attribute__track" aria-label={`${publicAttributeLabels[name]} ${Math.round(value * 100)}%`}>
                  <span style={{ width: `${value * 100}%` }} />
                </div>
              </div>
            ))}
        </div>
      </section>
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
