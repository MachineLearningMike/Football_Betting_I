from datapackage import Package

package = Package('https://datahub.io/sports-data/german-bundesliga/datapackage.json')

# print list of all resources:
print(package.resource_names)

# print processed tabular data (if exists any)
for resource in package.resources:
    if resource.descriptor['datahub']['type'] == 'derived/csv':
        data = resource.read()
        print( data )