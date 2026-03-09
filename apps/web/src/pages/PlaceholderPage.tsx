export default function PlaceholderPage({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <section className="placeholderPage" aria-label={title}>
      <div className="placeholderEyebrow">Coming later</div>
      <h1 className="placeholderTitle">{title}</h1>
      <p className="placeholderDescription">{description}</p>
    </section>
  );
}
