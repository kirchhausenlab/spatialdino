export default function PlaceholderPage({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <section className="placeholderPage" aria-label={title}>
      <h1 className="placeholderTitle">{title}</h1>
      <p className="placeholderDescription">{description}</p>
    </section>
  );
}
