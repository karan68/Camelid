/* Companion artifacts are catalog data supplied by the backend, never inferred
   from a model name in the UI. That keeps acquisition generic and makes the live
   disk scan the only authority for whether the complete bundle is installed. */

export function catalogCompanionArtifacts(item) {
  return Array.isArray(item?.companion_artifacts)
    ? item.companion_artifacts.filter((artifact) => artifact?.filename)
    : []
}

export function catalogArtifacts(item) {
  if (!item?.filename) return []
  return [
    {
      role: 'model',
      repo_id: item.repo_id,
      filename: item.filename,
      size_bytes: Number(item.size_bytes || 0),
    },
    ...catalogCompanionArtifacts(item),
  ]
}

export function missingCatalogArtifacts(item, localFilenames = new Set()) {
  return catalogArtifacts(item).filter((artifact) => !localFilenames.has(artifact.filename))
}

export function catalogBundleInstalled(item, localFilenames = new Set()) {
  const artifacts = catalogArtifacts(item)
  return artifacts.length > 0 && artifacts.every((artifact) => localFilenames.has(artifact.filename))
}

export function catalogDownloadBytes(item, localFilenames = new Set()) {
  return missingCatalogArtifacts(item, localFilenames)
    .reduce((total, artifact) => total + Number(artifact.size_bytes || 0), 0)
}
