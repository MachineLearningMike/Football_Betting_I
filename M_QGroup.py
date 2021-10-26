
class M_QGroup() :
    def __init__( self, q_group ) :
        self.q_group = q_group

    def GroupId_To_Queries( self, group_id ) :
        return [ group['Qs'] for key, group in self.q_group.items() ][ group_id ]

    def Query_To_GroupId( self, query_id ) :
        group_id = 0
        for g_name, group in self.q_group.items() :
            if query_id in group['Qs'] : break
            else : group_id += 1
        if group_id >= self.nGroups() :
            group_id = -1
        
        return group_id

    def Size_Splits( self ) :
        return [ len( group['Qs'] ) for key, group in self.q_group.items() ]
    
    def DummyQueries_GroupWise( self ) :
        return [ [d - min( group['Qs'] ) for d in group['dummy'] ] for key, group in self.q_group.items() ]

    def Dimensions_GroupWise( self ) :
        return [ len( group['Qs'] ) for key, group in self.q_group.items() ]

    def nGroups( self ) :
        return len( self.q_group )

    def Root_Queries( self ) :
        root_queries = []
        for key, value in self.q_group.items() : root_queries += value['roots']
        root_queries = list( set( root_queries ) ); root_queries.sort()
        return root_queries

    def Queries( self ) :
        queries = []
        for key, value in self.q_group.items() : 
            queries += value['Qs']
        for place in range( len( queries ) - 1 ) : assert queries[ place ] + 1 == queries[ place + 1]

        queries = list( set( queries ) ); queries.sort()

        return queries
