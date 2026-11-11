export default function SkillList({ title, skills, variant = 'neutral' }) {
  if (!skills?.length) return null;
  return (
    <div className="skills">
      <h4>{title}</h4>
      <ul aria-label={title}>
        {skills.map((s) => (
          <li key={s} className={`chip chip--${variant}`}>{s}</li>
        ))}
      </ul>
    </div>
  );
}
