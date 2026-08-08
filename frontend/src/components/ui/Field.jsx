import { cloneElement, isValidElement, useId } from 'react'

/* Field — labeled form control wrapper. Pass an <input>/<textarea>/<select> or children. */
const LABELABLE_TAGS = new Set(['input', 'textarea', 'select'])

export function Field({ label, hint, htmlFor, className = '', children }) {
  const generatedId = `cx-field-${useId()}`
  let control = children
  let labelFor = htmlFor

  if (typeof children === 'function') {
    labelFor = htmlFor || generatedId
    control = children(labelFor)
  } else if (isValidElement(children) && LABELABLE_TAGS.has(children.type)) {
    labelFor = children.props.id || htmlFor || generatedId
    if (!children.props.id) control = cloneElement(children, { id: labelFor })
  }

  return (
    <label className={`cx-field ${className}`.trim()} htmlFor={labelFor}>
      {label && <span className="cx-field__label">{label}</span>}
      {control}
      {hint && <span className="cx-field__hint">{hint}</span>}
    </label>
  )
}

export default Field
