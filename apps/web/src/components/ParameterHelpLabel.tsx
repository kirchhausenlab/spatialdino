type ParameterHelpLabelProps = {
  label: string;
  description: string;
};

export default function ParameterHelpLabel({ label, description }: ParameterHelpLabelProps) {
  return (
    <span className="parameterHelpLabel">
      <span>{label}</span>
      <span className="parameterHelpTrigger" tabIndex={0} aria-label={`${label}: ${description}`}>
        ?
        <span className="parameterHelpTooltip" role="tooltip">
          {description}
        </span>
      </span>
    </span>
  );
}
