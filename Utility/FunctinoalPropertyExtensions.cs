using System.Globalization;
using System.Reflection;

namespace ReadyDOS.Shared.Utility {
    public static class FunctinoalPropertyExtensions {
        /// <summary>
        /// Sets a public instance property by name if it exists and is writable.
        /// </summary>
        /// <param name="target">Target object to update.</param>
        /// <param name="propertyName">Property name to set.</param>
        /// <param name="value">Value to assign.</param>
        /// <returns><c>true</c> if the property existed and was set; otherwise <c>false</c>.</returns>
        public static readonly Func<object?, string, object?, bool> SetIfExists =
            (target, name, value) =>
                target is not null &&
                target.GetType().GetProperty(name, BindingFlags.Public | BindingFlags.Instance) is PropertyInfo property &&
                property.CanWrite &&
                (
                    value is null
                        ? (!property.PropertyType.IsValueType || Nullable.GetUnderlyingType(property.PropertyType) is not null) && Set(property, target, null)
                        : property.PropertyType.IsAssignableFrom(value.GetType()) && Set(property, target, value)
                );

        public static bool TrySet<T>(this T target, string propertyName, object? value) where T : class
            => SetIfExists(target, propertyName, value);

        private static bool Set(PropertyInfo p, object target, object? value) {
            p.SetValue(target, value);
            return true;
        }

        /// <summary>
        /// Reads a public instance property by name if it exists and is readable.
        /// No exceptions are used for control flow — the method is expression-based and functional.
        /// </summary>
        /// <typeparam name="T">Desired return type.</typeparam>
        /// <param name="source">Source object to read from.</param>
        /// <param name="propertyName">Property name to read.</param>
        /// <param name="fallback">Fallback value when missing or unreadable.</param>
        /// <returns>The value when available; otherwise <paramref name="fallback"/>.</returns>
        public static T? ReadIfExists<T>(this object? source, string propertyName, T? fallback) =>
            source is null
                ? fallback
                : source.GetType().GetProperty(propertyName, BindingFlags.Public | BindingFlags.Instance) is not PropertyInfo p || !p.CanRead
                    ? fallback
                    : p.GetValue(source) is T typed
                        ? typed
                        : TryConvert<T>(p.GetValue(source), out T? converted)
                            ? converted
                            : fallback;

        /// <summary>
        /// Attempts to convert a value into T using invariant culture.
        /// This function has no side effects and can be composed safely.
        /// </summary>
        private static bool TryConvert<T>(object? value, out T? result) {
            result = default;
            if (value is null)
                return false;

            try {
                Type dest = Nullable.GetUnderlyingType(typeof(T)) ?? typeof(T);
                object converted = Convert.ChangeType(value, dest, CultureInfo.InvariantCulture);
                result = (T)converted;
                return true;
            }
            catch {
                return false;
            }
        }
    }
}
