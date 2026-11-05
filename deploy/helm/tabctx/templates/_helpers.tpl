{{- define "tabctx.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "tabctx.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else if contains (include "tabctx.name" .) .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name (include "tabctx.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "tabctx.selectorLabels" -}}
app.kubernetes.io/name: {{ include "tabctx.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "tabctx.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{ include "tabctx.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "tabctx.tabpfnSecret" -}}
{{- if .Values.existingTabpfnSecret -}}
{{- .Values.existingTabpfnSecret -}}
{{- else if .Values.tabpfnToken -}}
{{- printf "%s-tabpfn" (include "tabctx.fullname" .) -}}
{{- end -}}
{{- end -}}

{{/* The claim a volume ("spill" or "weights") uses. */}}
{{- define "tabctx.claim" -}}
{{- $volume := index . 1 -}}
{{- $volume.existingClaim | default (printf "%s-%s" (include "tabctx.fullname" (index . 0)) (index . 2)) -}}
{{- end -}}

{{/* Settings the chart derives from its values, as name: value. */}}
{{- define "tabctx.env" -}}
TABCTX_BACKEND: {{ join "," .Values.models | quote }}
TABCTX_REQUIRE_TENANT: {{ .Values.requireTenant | quote }}
TABCTX_GPU_MEMORY_FRACTION: {{ .Values.gpuMemoryFraction | quote }}
TABCTX_MAX_ONGOING_REQUESTS: {{ .Values.maxOngoingRequests | quote }}
TABCTX_MAX_QUEUED_REQUESTS: {{ .Values.maxQueuedRequests | quote }}
TABCTX_OFFLOAD_DIR: "/tmp/offload"
{{- if .Values.spill.enabled }}
TABCTX_SPILL_DIR: "/data/spill"
# 90% of the volume.
TABCTX_SPILL_CAPACITY_BYTES: {{ mul (int .Values.spill.sizeGi) 966367641 | quote }}
{{- end }}
{{- if .Values.weightsCache.enabled }}
HF_HOME: "/cache/huggingface"
TABPFN_MODEL_CACHE_DIR: "/cache/tabpfn"
TORCHINDUCTOR_CACHE_DIR: "/cache/compile/inductor"
TRITON_CACHE_DIR: "/cache/compile/triton"
{{- end }}
{{- end -}}
