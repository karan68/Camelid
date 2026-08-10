import { groupByModelFamily } from '../../lib/modelFamilies.js'

/* Native disclosures keep the interaction keyboard-accessible and let each
   family remember its own open/closed state without a page-wide state machine. */
export function ModelFamilyGroups({
  items = [],
  renderItem,
  initiallyOpen = () => false,
  className = '',
}) {
  return groupByModelFamily(items).map((group) => (
    <details
      className={`model-family-group${className ? ` ${className}` : ''}`}
      key={group.family}
      open={initiallyOpen(group)}
    >
      <summary>
        <span className="model-family-group__name">{group.family}</span>
        <span className="model-family-group__count">
          {group.items.length} model{group.items.length === 1 ? '' : 's'}
        </span>
      </summary>
      <div className="model-family-group__items">
        {group.items.map(renderItem)}
      </div>
    </details>
  ))
}
